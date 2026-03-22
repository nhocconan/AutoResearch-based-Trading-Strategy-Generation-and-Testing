#!/usr/bin/env python3
"""
Experiment #265: 15m KAMA Trend with 4h HMA Bias and ADX Filter

Hypothesis: After analyzing 264 experiments, the clearest pattern is:
- RSI/mean-reversion strategies CONSISTENTLY FAIL on BTC/ETH (see #253-259, #261)
- Complex ensembles FAIL (see #256)
- Simple trend-following with strong HTF bias WORKS BEST (see #264 Sharpe=0.142)

Current BEST: mtf_4h_kama_1d_hma_adx_atr_v1 with Sharpe=0.478

For 15m timeframe, I'll use:
1. KAMA (Kaufman Adaptive Moving Average) - adapts to market efficiency, reduces whipsaw in chop
2. 4h HMA(21) - directional bias filter (proven in current best strategy)
3. ADX(14) > 25 - ensures we only trade in trending conditions
4. ATR ratio filter - avoid entries during vol spikes (ATR(7)/ATR(30) < 1.5)
5. Clean entry logic: KAMA crossover + HTF bias + ADX confirmation
6. Position sizing: 0.25-0.30 discrete, 2.5*ATR trailing stop

Why 15m with this setup:
- KAMA adapts to 15m noise better than EMA/SMA
- 4h HMA provides strong directional filter (tested in #264)
- ADX prevents choppy market entries (major cause of losses in #253, #259)
- Simple logic = fewer false signals, less fee drag
- Conservative sizing controls drawdown during 2022-style crashes

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_trend_4h_hma_adx_atr_v1"
timeframe = "15m"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - moves fast in trends, slow in chop.
    This is critical for 15m timeframe which has significant noise.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    # Calculate +DI and -DI using Wilder's smoothing
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / atr[i]
            minus_di[i] = 100 * minus_dm_s[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    dx_s = pd.Series(dx)
    adx_vals = dx_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_vals
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28  # Base position size (conservative for 15m)
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = strong directional bias (hard filter)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 25 = trending market (avoid chop)
        trending_market = adx[i] > 25
        
        # === VOLATILITY FILTER ===
        # Avoid entries during vol spikes (ATR ratio < 1.5)
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 999
        normal_volatility = atr_ratio < 1.5
        
        # === KAMA CROSSOVER SIGNALS ===
        # Fast KAMA crosses above slow KAMA = bullish momentum
        kama_bull_cross = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        
        # Fast KAMA crosses below slow KAMA = bearish momentum
        kama_bear_cross = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # === KAMA TREND CONFIRMATION ===
        # Price above both KAMAs = strong uptrend
        price_above_kama = close[i] > kama_fast[i] and close[i] > kama_slow[i]
        
        # Price below both KAMAs = strong downtrend
        price_below_kama = close[i] < kama_fast[i] and close[i] < kama_slow[i]
        
        # === DIRECTIONAL MOVEMENT CONFIRMATION ===
        # +DI > -DI = bullish directional movement
        di_bullish = plus_di[i] > minus_di[i]
        
        # -DI > +DI = bearish directional movement
        di_bearish = minus_di[i] > plus_di[i]
        
        # Determine position size based on volatility
        if not normal_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + KAMA cross + ADX trend + DI confirmation
        # Multiple confirmations reduce false signals on 15m
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            (kama_bull_cross or (price_above_kama and di_bullish)) and  # KAMA momentum
            trending_market and  # ADX confirms trend
            normal_volatility  # Not in vol spike
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            (kama_bear_cross or (price_below_kama and di_bearish)) and  # KAMA momentum
            trending_market and  # ADX confirms trend
            normal_volatility  # Not in vol spike
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === ADX DROPS BELOW THRESHOLD ===
        # Exit if market becomes choppy (ADX < 20)
        if in_position and adx[i] < 20:
            new_signal = 0.0  # Market no longer trending
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals