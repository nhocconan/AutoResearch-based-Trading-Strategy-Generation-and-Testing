#!/usr/bin/env python3
"""
Experiment #188: 30m EMA Crossover + 4h HMA Trend + Choppiness Regime + ADX Filter + ATR Stop

Hypothesis: 30m timeframe needs strong regime filtering to avoid whipsaws that killed
previous 30m strategies (#176, #182, #187 all had Sharpe < -1.0). Key insight:
- Choppiness Index detects range vs trend regime (CHOP > 61.8 = range, < 38.2 = trend)
- Only trade EMA crossovers when CHOP indicates trending regime
- 4h HMA provides stable higher-timeframe bias (proven in best strategy mtf_4h_kama_1d_hma_adx_atr_v1)
- Lower ADX threshold (15 instead of 25) for 30m to ensure sufficient trade count
- Conservative sizing (0.28) controls drawdown while allowing enough trades

Why this might work on 30m when others failed:
- #176 (KAMA): No regime filter → whipsawed in ranges
- #182 (Donchian): Breakout strategy fails without trend confirmation
- #187 (Supertrend): Supertrend is terrible in choppy markets
- This strategy ONLY trades when Choppiness Index confirms trending regime
- 4h HMA filter prevents counter-trend trades (major failure mode in crypto)

Learning from failures:
- Mean reversion fails on crypto (CRSI #181 Sharpe=-5.141)
- Pure trend following needs regime filter (Supertrend #187 Sharpe=-1.239)
- Donchian breakouts need trend confirmation (#178, #182 failed)
- Position sizing MUST be conservative (0.28 not 0.35+)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_4h_hma_chop_regime_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Market is choppy/ranging (avoid trend strategies)
    - CHOP < 38.2 = Market is trending (use trend strategies)
    - 38.2 to 61.8 = Transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar (simple TR, not smoothed)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Price range
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral
    
    # Fill initial values
    chop[:period] = 50.0
    
    return chop

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 50 = trending regime (allow trend trades)
        # CHOP > 55 = choppy regime (avoid trend trades, stay flat)
        trending_regime = chop[i] < 50.0
        choppy_regime = chop[i] > 55.0
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 30m to ensure trades)
        trend_strength = adx[i] > 15
        
        # === EMA CROSSOVER SIGNAL ===
        # EMA8 > EMA21 = bullish momentum
        # EMA8 < EMA21 = bearish momentum
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # EMA crossover detection (for entry timing)
        ema_cross_long = ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]
        ema_cross_short = ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]
        
        # === EMA STRUCTURE ===
        # EMA8 > EMA21 > EMA50 = strong bullish structure
        # EMA8 < EMA21 < EMA50 = strong bearish structure
        ema_structure_bull = ema_8[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_structure_bear = ema_8[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + trending regime + ADX strong + EMA bullish + (crossover OR structure)
        if bull_trend_4h and trending_regime and trend_strength and ema_bullish:
            # Entry on crossover OR strong structure + RSI confirmation
            if ema_cross_long or (ema_structure_bull and rsi_bullish):
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + trending regime + ADX strong + EMA bearish + (crossover OR structure)
        if bear_trend_4h and trending_regime and trend_strength and ema_bearish:
            # Entry on crossover OR strong structure + RSI confirmation
            if ema_cross_short or (ema_structure_bear and rsi_bearish):
                new_signal = -SIZE_BASE
        
        # === CHOPPY REGIME EXIT ===
        # If market becomes choppy, exit positions to avoid whipsaws
        if choppy_regime and in_position:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
            if position_side > 0:
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
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
            else:
                # Maintaining same position direction - update extremes
                if position_side > 0 and close[i] > highest_close:
                    highest_close = close[i]
                if position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                    lowest_close = close[i]
        else:
            # Exiting position (signal-based or stoploss or choppy regime)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals