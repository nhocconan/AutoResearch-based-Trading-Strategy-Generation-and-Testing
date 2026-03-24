#!/usr/bin/env python3
"""
Experiment #1040: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: The Ehlers Fisher Transform excels at identifying reversal points in bear/range markets
(2022 crash, 2025 bear) while Choppiness Index filters regime to avoid whipsaws. Combined with
1d/1w HMA trend bias, this creates a regime-adaptive strategy that works across all market conditions.

Key innovations for 6h timeframe:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, crosses at ±1.5 signal reversals
   - Proven to catch bear market rallies and crash bottoms better than RSI
2. Choppiness Index (14): CHOP>55 = range (fade Fisher extremes), CHOP<45 = trend (follow HMA)
3. 1d/1w HMA alignment: Only take long trades when 1d_HMA > 1w_HMA (and vice versa for shorts)
4. 6h-specific tuning: Fewer trades than 4h (30-60/year target), larger moves per trade
5. Asymmetric entries: In bear regime (price<1d_HMA<1w_HMA), only short retracements; in bull, only long dips

Why 6h should work:
- Captures multi-day swings without 4h noise or 12h slowness
- 6h bars = 4 per day, 28 per week — enough data for Fisher to stabilize
- Regime switching adapts to 2022 crash (trend mode) and 2025 range (mean reversion mode)
- Fisher Transform has superior reversal detection vs RSI in literature

Entry conditions (LOOSE to guarantee trades on all symbols):
- LONG range: CHOP>50 + Fisher<-1.5 + Fisher crossing up + price>1w_HMA*0.90
- LONG trend: CHOP<45 + price>1d_HMA>1w_HMA + Fisher>-1.0 + Fisher crossing up
- SHORT range: CHOP>50 + Fisher>+1.5 + Fisher crossing down + price<1w_HMA*1.10
- SHORT trend: CHOP<45 + price<1d_HMA<1w_HMA + Fisher<+1.0 + Fisher crossing down

Risk management:
- 2.5x ATR(14) trailing stoploss
- Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
- Position tracking for stoploss triggers

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_hma_regime_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * ((price - LL) / (HH - LL) - 0.5)
    
    Catches reversals better than RSI in bear/range markets
    Long signal: Fisher crosses above -1.5 from below
    Short signal: Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        x_raw = (close[i] - lowest_low) / price_range
        
        # Transform to -1 to +1 range with smoothing factor
        x = 0.66 * (x_raw - 0.5) + 0.67 * (0.66 * ((close[i-1] - lowest_low) / price_range - 0.5) if i > period else 0)
        x = max(-0.999, min(0.999, x))  # Clamp to avoid log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # Range market (slightly lower threshold for more trades)
        is_trending = chop_14[i] < 45.0  # Trend market
        
        # === HTF BIAS (1d/1w HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment (both 1d and 1w agree)
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # Fisher crossover signals
        fisher_cross_up = (not np.isnan(fisher_prev[i]) and 
                          fisher_prev[i] < -1.5 and fisher[i] >= -1.5)
        fisher_cross_down = (not np.isnan(fisher_prev[i]) and 
                            fisher_prev[i] > 1.5 and fisher[i] <= 1.5)
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - fade Fisher extremes
            # Long when Fisher extremely oversold + above 1w_HMA support
            if fisher_oversold and close[i] > hma_1w_aligned[i] * 0.92:
                desired_signal = SIZE_BASE
            # Stronger long on Fisher cross up from oversold
            elif fisher_cross_up and close[i] > hma_1w_aligned[i] * 0.90:
                desired_signal = SIZE_STRONG
            
            # Short when Fisher extremely overbought + below 1w_HMA resistance
            if fisher_overbought and close[i] < hma_1w_aligned[i] * 1.08:
                desired_signal = -SIZE_BASE
            # Stronger short on Fisher cross down from overbought
            elif fisher_cross_down and close[i] < hma_1w_aligned[i] * 1.10:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - follow HMA alignment with Fisher confirmation
            # Long in strong uptrend with Fisher not overbought
            if strong_bull and fisher[i] < 1.0 and fisher[i] > -1.5:
                # Enter on Fisher bounce from neutral
                if fisher[i] > fisher_prev[i] if not np.isnan(fisher_prev[i]) else True:
                    desired_signal = SIZE_STRONG
            
            # Short in strong downtrend with Fisher not oversold
            if strong_bear and fisher[i] > -1.0 and fisher[i] < 1.5:
                # Enter on Fisher drop from neutral
                if fisher[i] < fisher_prev[i] if not np.isnan(fisher_prev[i]) else True:
                    desired_signal = -SIZE_STRONG
            
            # Weaker trend signals (single HMA alignment)
            if hma_1d_bull and hma_1w_bull and fisher[i] < 0.5:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and fisher[i] > -0.5:
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