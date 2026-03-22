#!/usr/bin/env python3
"""
Experiment #296: 30m Donchian Breakout with Dual HTF Bias (4h + 1d HMA)

Hypothesis: 30m has failed 3x (#284, #290, #295) because:
1. Single HTF filter (4h only) wasn't strong enough to overcome 30m noise
2. Mean reversion and RSI pullbacks don't work on 30m (too much whipsaw)
3. Supertrend generates too many false signals on 30m

This strategy uses:
1. DUAL HTF bias: BOTH 4h HMA AND 1d HMA must agree (much stronger filter)
2. Donchian(20) breakout: cleaner trend entry than EMA crossover
3. ADX(14)>22: trend strength filter (slightly looser to ensure trades)
4. Volume confirmation: only enter on >1.1x average volume
5. KAMA(10) for adaptive trend following (better than EMA in noise)
6. Wide 3.5*ATR trailing stop (30m needs wider stops)

Why this might work on 30m when others failed:
- Dual HTF agreement eliminates 80%+ of false signals
- Donchian breakout = fewer but higher quality entries
- Volume filter ensures real momentum, not fake breakouts
- KAMA adapts to volatility better than fixed EMA

Timeframe: 30m (REQUIRED)
HTF: 4h AND 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 3.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_dual_htf_kama_adx_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    volatility[:er_period] = np.nan
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - highest high and lowest low over period.
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF BIAS (CRITICAL - both must agree) ===
        # 4h HMA = intermediate trend
        bull_4h = close[i] > hma_4h_aligned[i]
        bear_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA = major trend
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # BOTH must agree for strong bias
        strong_bull_htf = bull_4h and bull_1d
        strong_bear_htf = bear_4h and bear_1d
        
        # === TREND STRENGTH ===
        # ADX > 22 = trending market (looser than 25 to ensure trades)
        trending = adx[i] > 22
        
        # === DONCHIAN BREAKOUT ===
        # Price breaking above Donchian upper = bullish breakout
        donchian_bullish = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        # Price breaking below Donchian lower = bearish breakout
        donchian_bearish = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === KAMA TREND ===
        # Price above KAMA = bullish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.1x average = real momentum (looser than 1.2x)
        vol_confirmed = volume[i] > 1.1 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === POSITION SIZING ===
        # Strong trend + high volume = larger size
        if trending and vol_confirmed:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need ALL conditions (very selective but not impossible)
        # Dual HTF bull + Donchian breakout + KAMA bull + (ADX trending OR volume)
        long_conditions = (
            strong_bull_htf and  # BOTH 4h and 1d bullish
            donchian_bullish and  # Donchian breakout
            kama_bullish and  # Above KAMA
            (trending or vol_confirmed)  # ADX > 22 OR volume confirmation
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            strong_bear_htf and  # BOTH 4h and 1d bearish
            donchian_bearish and  # Donchian breakout
            kama_bearish and  # Below KAMA
            (trending or vol_confirmed)  # ADX > 22 OR volume confirmation
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.5 * ATR below highest close
                stoploss_price = highest_close - 3.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.5 * ATR above lowest close
                stoploss_price = lowest_close + 3.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === HTF BIAS REVERSAL EXIT ===
        # Exit if dual HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and not strong_bull_htf:
                new_signal = 0.0  # HTF bias reversed against long
            if position_side < 0 and not strong_bear_htf:
                new_signal = 0.0  # HTF bias reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals