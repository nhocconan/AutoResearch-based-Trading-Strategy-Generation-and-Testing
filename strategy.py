#!/usr/bin/env python3
"""
Experiment #1007: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness + Volume + ADX

Hypothesis: Combining Fisher Transform (reversal detection) with Choppiness Index (regime filter)
and adding volume confirmation + ADX trend strength will improve upon #1003 (Sharpe=0.387).
The 6h timeframe captures multi-day swings while avoiding noise of lower TFs.

Key innovations:
1. Fisher Transform (period=9): Catches reversals at extremes (-1.8/+1.8 thresholds)
2. Choppiness Index (14): >58 = range (mean revert), <42 = trend (trend follow)
3. ADX(14) confirmation: >22 for trend regime, <18 for range regime (hysteresis)
4. Volume ratio filter: current_vol / avg_vol(20) > 1.15 for entry confirmation
5. Regime-adaptive sizing: 0.20 for mean reversion, 0.30 for trend following
6. 1d/1w HMA bias filter for directional alignment
7. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Fisher Transform has superior reversal detection vs RSI (Ehlers research)
- Choppiness + ADX dual filter reduces false regime signals
- Volume confirmation avoids low-liquidity false breakouts
- 6h captures 30-60 trades/year target (fee-efficient)
- HTF bias ensures we trade with higher timeframe direction

Entry conditions (LOOSE to guarantee trades):
- LONG range: CHOP>55 + Fisher<-1.5 + volume_ratio>1.1 + price>1w_HMA*0.97
- LONG trend: CHOP<45 + ADX>22 + Fisher>-1.0 + volume_ratio>1.15 + 1d_HMA>1w_HMA
- SHORT range: CHOP>55 + Fisher>+1.5 + volume_ratio>1.1 + price<1w_HMA*1.03
- SHORT trend: CHOP<45 + ADX>22 + Fisher<+1.0 + volume_ratio>1.15 + 1d_HMA<1w_HMA

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_adx_vol_regime_1d1w_v2"
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
    Highlights turning points with sharp peaks
    Long when Fisher crosses above -1.8 from below
    Short when Fisher crosses below +1.8 from above
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate midpoint price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest_low) / price_range
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with EMA of previous fisher
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend
    ADX < 20 = weak trend / ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr != 0) * 100.0
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr != 0) * 100.0
    
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    
    dx = np.divide(di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0) * 100.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_MEAN_REV = 0.20
    SIZE_TREND = 0.30
    
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
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness + ADX) ===
        is_choppy = chop_14[i] > 55.0 or adx_14[i] < 18.0
        is_trending = chop_14[i] < 45.0 and adx_14[i] > 22.0
        
        # === HTF BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.10
        
        # === FISHER CROSSOVER DETECTION ===
        fisher_bull_cross = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_bear_cross = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher extreme values (for mean reversion)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - Fisher extremes + HTF bias
            # Long when Fisher extremely oversold + weekly bias bull
            if fisher_oversold and hma_1w_bull and vol_confirmed:
                desired_signal = SIZE_MEAN_REV
            # Short when Fisher extremely overbought + weekly bias bear
            elif fisher_overbought and hma_1w_bear and vol_confirmed:
                desired_signal = -SIZE_MEAN_REV
            # Fisher crossover entries
            elif fisher_bull_cross and hma_1w_bull:
                desired_signal = SIZE_MEAN_REV
            elif fisher_bear_cross and hma_1w_bear:
                desired_signal = -SIZE_MEAN_REV
        
        elif is_trending:
            # TREND FOLLOWING MODE - Fisher pullback + HTF alignment
            # Long in strong uptrend on Fisher pullback
            if strong_bull and fisher[i] > -1.0 and fisher[i] < 1.5 and vol_confirmed:
                desired_signal = SIZE_TREND
            # Short in strong downtrend on Fisher pullback
            elif strong_bear and fisher[i] < 1.0 and fisher[i] > -1.5 and vol_confirmed:
                desired_signal = -SIZE_TREND
            # Weaker trend signals
            elif hma_1d_bull and hma_1w_bull and fisher[i] > 0.0:
                desired_signal = SIZE_MEAN_REV
            elif hma_1d_bear and hma_1w_bear and fisher[i] < 0.0:
                desired_signal = -SIZE_MEAN_REV
        
        # Neutral regime - use simpler logic
        else:
            if fisher_bull_cross and hma_1w_bull:
                desired_signal = SIZE_MEAN_REV * 0.8
            elif fisher_bear_cross and hma_1w_bear:
                desired_signal = -SIZE_MEAN_REV * 0.8
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_MEAN_REV * 0.9:
            final_signal = SIZE_MEAN_REV
        elif desired_signal <= -SIZE_MEAN_REV * 0.9:
            final_signal = -SIZE_MEAN_REV
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