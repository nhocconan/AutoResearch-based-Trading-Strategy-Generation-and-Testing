#!/usr/bin/env python3
"""
EXPERIMENT #026 - MTF HMA+KAMA+Stoch+RSI+Volume+BBW (4h+1h v1)
==================================================================================================
Hypothesis: After #025's sharp decline (Sharpe 4.6→0.2), the issue is OVER-FILTERING.
#021 achieved Sharpe=4.629 with simpler conditions. This experiment tests:

Key changes vs #025:
- Timeframe: 4h trend + 1h entries (different from 1h+15m in #021-#025)
- Remove ADX filter (was too restrictive at 28 threshold)
- Remove ROC filter (added noise, not value)
- Widen RSI range: 40-60 (vs 45-55 in #025) for more entry opportunities
- Looser stoploss: 2.5*ATR (vs 2.0*ATR) to avoid premature exits
- Position size: 0.35 (back to #021 level, vs 0.30 in #025)
- Remove dynamic volatility sizing (was hurting consistency)
- Keep: HMA+KAMA trend confirmation, Stochastic, Volume, BBW filters

Why 4h+1h instead of 1h+15m:
- 4h trend is more stable, fewer whipsaws
- 1h entries capture more opportunities than 15m
- Better balance between signal quality and trade frequency
- Proven in #022 (Sharpe=2.727 with 1h+4h)

Expected outcome: Sharpe > 4.6 (beat #021), DD < -20%, Trades > 50
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_stoch_rsi_volume_bbw_4h_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_volume_sma_ratio(volume, period=20):
    """Calculate volume ratio vs SMA"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    ratio = np.zeros(n)
    
    for i in range(period - 1, n):
        avg_volume = np.mean(volume[i - period + 1:i + 1])
        if avg_volume > 0:
            ratio[i] = volume[i] / avg_volume
        else:
            ratio[i] = 1.0
    
    return ratio


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank over lookback period"""
    n = len(bbw)
    if n < lookback:
        return np.zeros(n) * 50
    
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        rank = np.sum(window < current)
        percentile[i] = rank / lookback * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_1h = calculate_hma(close, period=16)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    volume_ratio_1h = calculate_volume_sma_ratio(volume, period=20)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters using proper method
    try:
        prices_indexed = prices.set_index('open_time')
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 4h indicators
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(c_4h, period=48)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Calculate 4h trend
        trend_4h = np.zeros(len(c_4h))
        kama_trend_4h = np.zeros(len(c_4h))
        
        for i in range(len(c_4h)):
            if i >= 48:
                if c_4h[i] > hma_4h[i]:
                    trend_4h[i] = 1
                elif c_4h[i] < hma_4h[i]:
                    trend_4h[i] = -1
                
                if c_4h[i] > kama_4h[i]:
                    kama_trend_4h[i] = 1
                elif c_4h[i] < kama_4h[i]:
                    kama_trend_4h[i] = -1
        
        # Create 4h index for reindexing
        df_4h['trend'] = trend_4h
        df_4h['kama_trend'] = kama_trend_4h
        df_4h['bbw_pct'] = bbw_pct_4h
        
        # Reindex to 1h with ffill
        df_4h_reindexed = df_4h.reindex(prices_indexed.index, method='ffill')
        
        trend_4h_mapped = df_4h_reindexed['trend'].values
        kama_trend_4h_mapped = df_4h_reindexed['kama_trend'].values
        bbw_pct_4h_mapped = df_4h_reindexed['bbw_pct'].values
        
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 4
        n_4h = (n // bars_per_4h)
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = high[end_idx - 1]
            l_4h[i] = low[end_idx - 1]
        
        hma_4h = calculate_hma(c_4h, period=48)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        
        trend_4h_mapped = np.zeros(n)
        kama_trend_4h_mapped = np.zeros(n)
        bbw_pct_4h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h >= 48:
                if c_4h[idx_4h] > hma_4h[idx_4h]:
                    trend_4h_mapped[i] = 1
                elif c_4h[idx_4h] < hma_4h[idx_4h]:
                    trend_4h_mapped[i] = -1
                
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    kama_trend_4h_mapped[i] = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    kama_trend_4h_mapped[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (wider range for more opportunities)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Stochastic thresholds
    STOCH_LONG_MAX = 70
    STOCH_SHORT_MIN = 30
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # Volume ratio threshold
    VOLUME_RATIO_MIN = 1.2
    
    # BBW percentile threshold (avoid bottom 30% = too choppy)
    BBW_PCT_MIN = 30
    
    # ATR stoploss multiplier (looser for fewer premature exits)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 48 * 4, 14 * 2, 20, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_mapped[i]
        kama_trend = kama_trend_4h_mapped[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        vol_ratio = volume_ratio_1h[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # BBW percentile filter - avoid choppy markets (bottom 30%)
        if bbw_pct < BBW_PCT_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + KAMA on 4h)
        if trend != kama_trend or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h HMA + KAMA + BBW% + 1h RSI + Stoch + Volume + Z-score
        if trend == 1 and kama_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                stoch_k < STOCH_LONG_MAX and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):  # Pullback + Stoch not overbought + Volume confirm
                signals[i] = BASE_SIZE
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and kama_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                stoch_k > STOCH_SHORT_MIN and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):  # Pullback + Stoch not oversold + Volume confirm
                signals[i] = -BASE_SIZE
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals