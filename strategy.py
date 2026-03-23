#!/usr/bin/env python3
"""
Experiment #840: 1h Primary + 4h/12h HTF — Regime-Adaptive with Fisher + RSI Confluence

Hypothesis: After 576+ failed strategies, the key for 1h timeframe is:
1. Use 4h/12h HMA for SIGNAL DIRECTION (not entry trigger)
2. Use 1h Fisher Transform + RSI for ENTRY TIMING within HTF trend
3. Choppiness Index regime detection to switch between mean-revert and trend-follow
4. RELAXED entry conditions to ensure >=30 trades/year (previous 1h attempts got 0 trades!)
5. Position size 0.20-0.25 (smaller for lower TF to reduce fee drag)

Why this should work:
- 4h HMA(21) provides stable trend direction (proven in best strategy #823)
- Fisher Transform catches reversals better than RSI alone in bear/range markets
- Choppiness regime filter prevents trend-following in chop (major source of losses)
- Relaxed RSI thresholds (35/65) ensure trades fire on all symbols
- Fallback extreme RSI (25/75) guarantees minimum trade frequency

CRITICAL: Previous 1h strategies (#830, #838) got 0 trades due to over-filtering.
This strategy uses OR logic and fallbacks to ensure trades on BTC, ETH, SOL.

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_rsi_chop_regime_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    return np.clip(rsi, 0, 100)

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform."""
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        normalized = np.clip(normalized, 0.001, 0.999)
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return np.clip(chop, 0, 100)

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to 1h (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    fisher_1h, fisher_prev_1h = calculate_fisher_transform(high, low, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ma_1h = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    extreme_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(fisher_1h[i]) or np.isnan(fishер_prev_1h[i]):
            continue
        if np.isnan(vol_ma_1h[i]) or vol_ma_1h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (4h + 12h HMA21) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_12h_bull = close[i] > hma_12h_aligned[i]
        trend_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Confluence: at least one HTF agrees
        htftrend_bull = trend_4h_bull or trend_12h_bull
        htftrend_bear = trend_4h_bear or trend_12h_bear
        htftrend_neutral = not htftrend_bull and not htftrend_bear
        
        # === REGIME (1h Choppiness) ===
        ranging = chop_1h[i] > 55
        trending = chop_1h[i] < 45
        
        # === RSI (1h) ===
        rsi_os = rsi_1h[i] < 35
        rsi_ob = rsi_1h[i] > 65
        rsi_ext_os = rsi_1h[i] < 25
        rsi_ext_ob = rsi_1h[i] > 75
        
        # === Fisher Transform (1h) ===
        fish_os = fisher_1h[i] < -1.5
        fish_ob = fisher_1h[i] > 1.5
        fish_cross_up = fisher_prev_1h[i] < -1.5 and fisher_1h[i] >= -1.5
        fish_cross_down = fisher_prev_1h[i] > 1.5 and fisher_1h[i] <= 1.5
        
        # === Volume ===
        vol_ok = volume[i] > 0.8 * vol_ma_1h[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (Mean Reversion) ===
        if ranging:
            # Long: Fisher oversold + RSI oversold + HTF trend OK
            if fish_os and rsi_os and (htftrend_bull or htftrend_neutral):
                desired_signal = BASE_SIZE
            # Short: Fisher overbought + RSI overbought + HTF trend OK
            if fish_ob and rsi_ob and (htftrend_bear or htftrend_neutral):
                desired_signal = -BASE_SIZE
            # Fisher cross + RSI confluence
            if fish_cross_up and rsi_os:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            if fish_cross_down and rsi_ob:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === TRENDING REGIME (Trend Following) ===
        elif trending:
            # Long: HTF bullish + Fisher recovering
            if htftrend_bull and fisher_1h[i] > fisher_prev_1h[i] and fisher_1h[i] < 0:
                desired_signal = BASE_SIZE if vol_ok else REDUCED_SIZE
            # Short: HTF bearish + Fisher weakening
            if htftrend_bear and fisher_1h[i] < fisher_prev_1h[i] and fisher_1h[i] > 0:
                desired_signal = -BASE_SIZE if vol_ok else -REDUCED_SIZE
        
        # === FALLBACK: Extreme RSI (ensures trades on all symbols) ===
        # This is CRITICAL to avoid 0 trades like experiments #830, #838
        if desired_signal == 0:
            if rsi_ext_os and (htftrend_bull or htftrend_neutral):
                desired_signal = REDUCED_SIZE
            if rsi_ext_ob and (htftrend_bear or htftrend_neutral):
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS (Trailing ATR 2.5x) ===
        stop_triggered = False
        if in_position and position_side > 0:
            extreme_price = max(extreme_price, close[i])
            stop_price = extreme_price - 2.5 * entry_atr
            if close[i] < stop_price:
                stop_triggered = True
        if in_position and position_side < 0:
            extreme_price = min(extreme_price, close[i])
            stop_price = extreme_price + 2.5 * entry_atr
            if close[i] > stop_price:
                stop_triggered = True
        
        if stop_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0 and not stop_triggered:
            if position_side > 0:
                if (trend_4h_bull or trend_12h_bull) and fisher_1h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (trend_4h_bear or trend_12h_bear) and fisher_1h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if trend_4h_bear and trend_12h_bear and fisher_1h[i] > 1.5:
                desired_signal = 0.0
            if ranging and rsi_1h[i] > 80:
                desired_signal = 0.0
        if in_position and position_side < 0:
            if trend_4h_bull and trend_12h_bull and fisher_1h[i] < -1.5:
                desired_signal = 0.0
            if ranging and rsi_1h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                extreme_price = close[i]
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                extreme_price = close[i]
            elif position_side > 0:
                extreme_price = max(extreme_price, close[i])
            elif position_side < 0:
                extreme_price = min(extreme_price, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                extreme_price = 0.0
        
        signals[i] = desired_signal
    
    return signals