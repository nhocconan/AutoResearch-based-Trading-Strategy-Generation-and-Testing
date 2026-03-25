#!/usr/bin/env python3
"""
Experiment #1135: 6h Primary + 12h/1d HTF — Fisher Transform + Regime-Adaptive RSI

Hypothesis: 6h timeframe is underexplored (0 experiments). Using Ehlers Fisher Transform
for reversal detection combined with regime-adaptive logic (ADX filter) and 12h/1d HTF
bias will outperform pure trend-following on 6h. Fisher Transform normalizes price
into Gaussian distribution, making extremes more reliable for reversals.

Key innovations:
1. Ehlers Fisher Transform (period=9): Converts price to bounded -1.5 to +1.5 range
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. ADX(14) regime filter: <25 = range (use Fisher reversals), >25 = trend (use HMA)
3. 12h HMA(21) bias: Only long if 12h HMA sloping up, only short if sloping down
4. 1d ADX confirmation: Avoid entries when 1d ADX > 40 (extreme trend = reversal risk)
5. RSI(7) secondary filter: Confirms Fisher signals (RSI<35 for long, RSI>65 for short)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- Fisher Transform catches reversals better than RSI in bear/range markets (2022-2025)
- 6h captures multi-day swings without 4h noise or 12h slowness
- Regime switching avoids trend-following whipsaws in 2022-2023 choppy markets
- 12h HMA provides intermediate trend bias without 1w being too slow
- Target: 30-50 trades/year (appropriate for 6h timeframe)

Entry conditions (LOOSE to guarantee trades):
- LONG reversal: ADX<25 + Fisher crosses -1.5 + RSI<40 + 12h_HMA sloping up
- LONG trend: ADX>25 + price>12h_HMA + 12h_HMA>prev + RSI>45
- SHORT reversal: ADX<25 + Fisher crosses +1.5 + RSI>60 + 12h_HMA sloping down
- SHORT trend: ADX>25 + price<12h_HMA + 12h_HMA<prev + RSI<55

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_regime_rsi_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Smooth using Wilder's method (EMA with span=period)
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.divide(plus_dm_smooth, atr_smooth, out=np.zeros_like(plus_dm_smooth), where=atr_smooth != 0) * 100
    minus_di = np.divide(minus_dm_smooth, atr_smooth, out=np.zeros_like(minus_dm_smooth), where=atr_smooth != 0) * 100
    
    # Calculate DX
    di_sum = plus_di + minus_di
    dx = np.divide(np.abs(plus_di - minus_di), di_sum, out=np.zeros_like(plus_di), where=di_sum != 0) * 100
    
    # ADX is EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution
    Bounded between approximately -1.5 and +1.5
    Crosses above -1.5 = oversold reversal signal
    Crosses below +1.5 = overbought reversal signal
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate (2 * (close - LL) / (HH - LL) - 1)
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        normalized = 2.0 * (close[i] - lowest_low) / price_range - 1.0
        
        # Clamp to avoid division issues
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous value for cross detection
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_12h_prev_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_prev_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_prev_raw)
    # Shift by 1 for previous value
    hma_12h_prev_aligned = np.roll(hma_12h_prev_aligned, 1)
    hma_12h_prev_aligned[0] = np.nan
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX) ===
        is_ranging = adx_14[i] < 25.0  # Range market - use Fisher reversals
        is_trending = adx_14[i] >= 25.0  # Trend market - use HMA trend follow
        
        # === HTF BIAS (12h HMA slope) ===
        hma_12h_slope_up = hma_12h_aligned[i] > hma_12h_prev_aligned[i] if not np.isnan(hma_12h_prev_aligned[i]) else False
        hma_12h_slope_down = hma_12h_aligned[i] < hma_12h_prev_aligned[i] if not np.isnan(hma_12h_prev_aligned[i]) else False
        
        price_above_hma = close[i] > hma_12h_aligned[i]
        price_below_hma = close[i] < hma_12h_aligned[i]
        
        # === 1D ADX FILTER (avoid extreme trends) ===
        extreme_trend = not np.isnan(adx_1d_aligned[i]) and adx_1d_aligned[i] > 40.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # Fisher cross detection
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5 if not np.isnan(fisher_prev[i]) else False
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5 if not np.isnan(fisher_prev[i]) else False
        
        if is_ranging and not extreme_trend:
            # MEAN REVERSION MODE - use Fisher Transform reversals
            # Long when Fisher crosses above -1.5 (oversold)
            if fisher_cross_up and rsi_7[i] < 40.0 and hma_12h_slope_up:
                desired_signal = SIZE_BASE
            # Stronger long signal
            elif fisher_cross_up and rsi_7[i] < 30.0 and hma_12h_slope_up:
                desired_signal = SIZE_STRONG
            
            # Short when Fisher crosses below +1.5 (overbought)
            elif fisher_cross_down and rsi_7[i] > 60.0 and hma_12h_slope_down:
                desired_signal = -SIZE_BASE
            # Stronger short signal
            elif fisher_cross_down and rsi_7[i] > 70.0 and hma_12h_slope_down:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use HMA alignment
            # Long in uptrend with RSI confirmation
            if price_above_hma and hma_12h_slope_up and rsi_14[i] > 45.0 and rsi_14[i] < 75.0:
                desired_signal = SIZE_STRONG
            # Short in downtrend with RSI confirmation
            elif price_below_hma and hma_12h_slope_down and rsi_14[i] < 55.0 and rsi_14[i] > 25.0:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif price_above_hma and hma_12h_slope_up and rsi_14[i] > 50.0:
                desired_signal = SIZE_BASE
            elif price_below_hma and hma_12h_slope_down and rsi_14[i] < 50.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals