#!/usr/bin/env python3
"""
EXPERIMENT #025 - MTF HMA+KAMA+Stoch+RSI+ADX+Volume+BBW+ROC (15m+1h v5)
==================================================================================================
Hypothesis: Building on #021 (Sharpe=4.629) and #024 (Sharpe=3.881), this experiment
optimizes the 15m+1h MTF approach with key refinements:

Key improvements vs #024:
- Timeframe: 15m entries + 1h trend (proven best combination from #021)
- Position size: 0.30 (reduced from 0.35 for better risk control)
- RSI range: 45-55 (tighter than 40-60 for higher quality pullback entries)
- Stochastic: (14,3,3) with 30/70 thresholds (tighter than 25/75)
- ADX(14) trend strength filter: must be > 28 (higher than 25 for stronger trends)
- Stoploss: 2.0*ATR (tighter than 2.5*ATR for better risk/reward)
- Volume ratio: 1.3x (higher than 1.2x for stronger confirmation)
- BBW percentile: > 35 (higher than 30 to avoid more choppy markets)
- HMA periods: 16/48 (same as #024, proven effective)
- Add ROC(10) momentum filter: must align with trend direction
- Add dynamic position sizing: base_size * (target_vol / current_vol)

Why this should beat #021 and #024:
- Tighter RSI range (45-55) captures only deepest pullbacks in trends
- Higher ADX threshold (28) ensures only strongest trends are traded
- Tighter stoploss (2.0*ATR) improves risk/reward ratio
- ROC momentum filter adds confirmation of trend continuation
- Dynamic position sizing reduces exposure during high volatility
- Higher volume threshold (1.3x) filters weaker moves
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_stoch_rsi_adx_volume_bbw_roc_15m_v5"
timeframe = "15m"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    tr_smooth[period - 1] = np.sum(tr[1:period])
    plus_dm_smooth[period - 1] = np.sum(plus_dm[1:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    adx[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    
    for i in range(period * 2 - 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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


def calculate_roc(close, period=10):
    """Calculate Rate of Change"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    roc = np.zeros(n)
    
    for i in range(period, n):
        if close[i - period] > 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
        else:
            roc[i] = 0
    
    return roc


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=16)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    volume_ratio_15m = calculate_volume_sma_ratio(volume, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    roc_15m = calculate_roc(close, period=10)
    
    # Resample to 1h for trend filters using proper method
    try:
        prices_indexed = prices.set_index('open_time')
        df_1h = prices_indexed.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 1h indicators
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        hma_1h = calculate_hma(c_1h, period=48)
        kama_1h = calculate_kama(c_1h, er_period=10, fast_period=2, slow_period=30)
        adx_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
        _, _, _, bbw_1h = calculate_bollinger_bands(c_1h, period=20, std_mult=2.0)
        bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
        
        # Calculate 1h trend
        trend_1h = np.zeros(len(c_1h))
        kama_trend_1h = np.zeros(len(c_1h))
        
        for i in range(len(c_1h)):
            if i >= 48:
                if c_1h[i] > hma_1h[i]:
                    trend_1h[i] = 1
                elif c_1h[i] < hma_1h[i]:
                    trend_1h[i] = -1
                
                if c_1h[i] > kama_1h[i]:
                    kama_trend_1h[i] = 1
                elif c_1h[i] < kama_1h[i]:
                    kama_trend_1h[i] = -1
        
        # Create 1h index for reindexing
        df_1h['trend'] = trend_1h
        df_1h['kama_trend'] = kama_trend_1h
        df_1h['adx'] = adx_1h
        df_1h['bbw_pct'] = bbw_pct_1h
        
        # Reindex to 15m with ffill
        df_1h_reindexed = df_1h.reindex(prices_indexed.index, method='ffill')
        
        trend_1h_mapped = df_1h_reindexed['trend'].values
        kama_trend_1h_mapped = df_1h_reindexed['kama_trend'].values
        adx_1h_mapped = df_1h_reindexed['adx'].values
        bbw_pct_1h_mapped = df_1h_reindexed['bbw_pct'].values
        
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_1h = 4
        n_1h = (n // bars_per_1h)
        
        c_1h = np.zeros(n_1h)
        h_1h = np.zeros(n_1h)
        l_1h = np.zeros(n_1h)
        for i in range(n_1h):
            start_idx = i * bars_per_1h
            end_idx = start_idx + bars_per_1h
            c_1h[i] = close[end_idx - 1]
            h_1h[i] = high[end_idx - 1]
            l_1h[i] = low[end_idx - 1]
        
        hma_1h = calculate_hma(c_1h, period=48)
        kama_1h = calculate_kama(c_1h, er_period=10, fast_period=2, slow_period=30)
        adx_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
        
        trend_1h_mapped = np.zeros(n)
        kama_trend_1h_mapped = np.zeros(n)
        adx_1h_mapped = np.zeros(n)
        bbw_pct_1h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_1h = i // bars_per_1h
            if idx_1h < n_1h and idx_1h >= 48:
                if c_1h[idx_1h] > hma_1h[idx_1h]:
                    trend_1h_mapped[i] = 1
                elif c_1h[idx_1h] < hma_1h[idx_1h]:
                    trend_1h_mapped[i] = -1
                
                if c_1h[idx_1h] > kama_1h[idx_1h]:
                    kama_trend_1h_mapped[i] = 1
                elif c_1h[idx_1h] < kama_1h[idx_1h]:
                    kama_trend_1h_mapped[i] = -1
                
                adx_1h_mapped[i] = adx_1h[idx_1h] if idx_1h < len(adx_1h) else 0
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries (tighter range for cleaner entries)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 55
    
    # Stochastic thresholds (tighter than 25/75)
    STOCH_LONG_MAX = 70
    STOCH_SHORT_MIN = 30
    
    # ADX threshold for trend strength (higher than 25)
    ADX_MIN = 28
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 1.8
    
    # Volume ratio threshold (higher than 1.2x)
    VOLUME_RATIO_MIN = 1.3
    
    # BBW percentile threshold (avoid bottom 35% = too choppy)
    BBW_PCT_MIN = 35
    
    # ATR stoploss multiplier (tighter for better R:R)
    ATR_STOP_MULT = 2.0
    
    # ROC threshold for momentum confirmation
    ROC_MIN = 0.5
    
    # Volatility target for dynamic sizing (annualized ~20%)
    TARGET_VOL = 0.02
    
    first_valid = max(200, 48 * 4, 14 * 2, 20, 28, 100, 14 * 2 * 4)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1h_mapped[i]
        kama_trend = kama_trend_1h_mapped[i]
        adx_val = adx_1h_mapped[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        vol_ratio = volume_ratio_15m[i]
        bbw_pct = bbw_pct_1h_mapped[i]
        roc_val = roc_15m[i]
        
        # Calculate dynamic position size based on volatility
        current_vol_pct = atr / price if price > 0 else TARGET_VOL
        if current_vol_pct > 0:
            vol_adjustment = min(1.5, max(0.5, TARGET_VOL / current_vol_pct))
        else:
            vol_adjustment = 1.0
        
        size_full = BASE_SIZE * vol_adjustment
        size_half = size_full / 2
        
        # BBW percentile filter - avoid choppy markets (bottom 35%)
        if bbw_pct < BBW_PCT_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + KAMA on 1h)
        if trend != kama_trend or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ADX trend strength filter - only trade strong trends
        if adx_val < ADX_MIN:
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
            
            # Stoploss check (2.0*ATR)
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
                    signals[i] = size_half
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
                    signals[i] = -size_half
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
        
        # Entry logic: 1h HMA + KAMA + ADX + BBW% + 15m RSI + Stoch + Volume + Z-score + ROC
        if trend == 1 and kama_trend == 1:  # Bullish trend confirmed on 1h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                stoch_k < STOCH_LONG_MAX and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN and
                roc_val > ROC_MIN):  # Pullback + Stoch not overbought + Volume + Momentum confirm
                signals[i] = size_full
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and kama_trend == -1:  # Bearish trend confirmed on 1h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                stoch_k > STOCH_SHORT_MIN and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN and
                roc_val < -ROC_MIN):  # Pullback + Stoch not oversold + Volume + Momentum confirm
                signals[i] = -size_full
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals