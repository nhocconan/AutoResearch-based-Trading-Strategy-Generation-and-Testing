#!/usr/bin/env python3
"""
EXPERIMENT #007 - MTF HMA Trend + Stochastic Entry + Volume + ATR Regime (4h+1h v1)
==================================================================================================
Hypothesis: Current best uses 4h DEMA+Supertrend+MACD. Let's try HMA (Hull MA) which is faster
and smoother than EMA, combined with Stochastic oscillator for entry timing (different from RSI/MACD).
Add volume confirmation to filter false breakouts and ATR regime filter to avoid extreme volatility.

Key changes from #006:
- Timeframe: 1h entries + 4h trend (proven MTF combo)
- Trend: HMA(16) vs HMA(48) crossover on 4h (faster than KAMA, smoother than EMA)
- Entry: Stochastic(14,3,3) cross on 1h (different from RSI/MACD - catches momentum shifts)
- Filter: Volume > 1.5x 20-period average + ATR regime (not extreme)
- Position size: 0.30 (discrete levels: 0.0, ±0.20, ±0.30)
- Stoploss: 2.0*ATR trailing, TP at 2R reduce to half

Why this should work:
- HMA reduces lag significantly vs EMA while staying smooth
- Stochastic catches momentum shifts better than RSI in ranging markets
- Volume confirmation filters false breakouts (institutional participation)
- ATR regime filter avoids trading during extreme volatility (reduces whipsaws)
- 4h trend filter reduces false signals vs 1h-only strategies
"""

import numpy as np
import pandas as pd

name = "mtf_hma_stoch_volume_atr_regime_4h_1h_v1"
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


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA for period, period/2, and sqrt(period)
    def wma(data, wma_period):
        n_data = len(data)
        result = np.zeros(n_data)
        weights = np.arange(1, wma_period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(wma_period - 1, n_data):
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        
        return result
    
    wma_period = wma(close, period)
    wma_half = wma(close, half_period)
    
    # Raw HMA = 2*WMA(period/2) - WMA(period)
    raw_hma = 2 * wma_half - wma_period
    
    # Final HMA = WMA of Raw HMA with sqrt(period)
    hma = wma(raw_hma, sqrt_period)
    
    return hma


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period + d_period:
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
    
    # %D is SMA of %K
    for i in range(k_period + d_period - 2, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = np.zeros(n)
    
    for i in range(period - 1, n):
        volume_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return volume_sma


def calculate_atr_regime(atr, period=50):
    """Calculate ATR regime (percentile of recent ATR values)"""
    n = len(atr)
    if n < period:
        return np.zeros(n)
    
    regime = np.zeros(n)
    
    for i in range(period - 1, n):
        window = atr[i - period + 1:i + 1]
        current_atr = atr[i]
        
        # Calculate percentile rank
        percentile = np.sum(window <= current_atr) / period
        
        # Regime: 0=low vol, 1=normal, 2=high vol
        if percentile < 0.2:
            regime[i] = 0  # Low volatility
        elif percentile > 0.8:
            regime[i] = 2  # High volatility
        else:
            regime[i] = 1  # Normal volatility
    
    return regime


def resample_to_higher_tf(prices, tf='4h'):
    """Resample prices to higher timeframe using actual timestamps"""
    prices_indexed = prices.set_index('open_time')
    
    df_resampled = prices_indexed.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Resample to 4h for trend filters
    prices_df = prices.copy()
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'])
    
    try:
        df_4h = resample_to_higher_tf(prices_df, '4h')
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 4  # 4 x 1h = 4h
        n_4h = n // bars_per_4h
        
        c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
        h_4h = np.array([np.max(high[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        l_4h = np.array([np.min(low[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        v_4h = np.array([np.sum(volume[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        
        df_4h = pd.DataFrame({
            'open': c_4h,
            'high': h_4h,
            'low': l_4h,
            'close': c_4h,
            'volume': v_4h
        })
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    atr_regime_1h = calculate_atr_regime(atr_1h, period=50)
    
    # 4h indicators for trend
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    
    # Map 4h indicators back to 1h timeframe using ffill
    hma_trend_4h = np.zeros(n)
    
    for i in range(n):
        # Find which 4h bar this 1h bar belongs to
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h >= 48:  # Need enough data for HMA(48)
            # HMA trend: fast HMA above/below slow HMA
            if hma_fast_4h[idx_4h] > hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = 1
            elif hma_fast_4h[idx_4h] < hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = -1
    
    # Entry thresholds
    STOCH_LONG_CROSS = 20  # %K crosses above %D below 20
    STOCH_SHORT_CROSS = 80  # %K crosses below %D above 80
    VOLUME_MULT = 1.5
    ATR_REGIME_OK = [0, 1]  # Only trade in low/normal volatility
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 48 * 4, 14 + 3, 20, 50)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(stoch_k_1h[i]) or np.isnan(stoch_d_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        hma_trend = hma_trend_4h[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        stoch_k_prev = stoch_k_1h[i - 1] if i > 0 else stoch_k
        stoch_d_prev = stoch_d_1h[i - 1] if i > 0 else stoch_d
        vol_ratio = volume[i] / volume_sma_1h[i] if volume_sma_1h[i] > 0 else 0
        atr_regime = atr_regime_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
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
        
        # Entry logic: 4h HMA trend + 1h Stochastic cross + Volume + ATR regime
        # Long entry: HMA bullish + Stochastic cross up from oversold + Volume confirmation + Normal ATR
        if hma_trend == 1 and atr_regime in ATR_REGIME_OK:
            stoch_cross_up = (stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d and stoch_k < STOCH_LONG_CROSS)
            volume_confirmed = vol_ratio >= VOLUME_MULT
            
            if stoch_cross_up and volume_confirmed:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: HMA bearish + Stochastic cross down from overbought + Volume confirmation + Normal ATR
        elif hma_trend == -1 and atr_regime in ATR_REGIME_OK:
            stoch_cross_down = (stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d and stoch_k > STOCH_SHORT_CROSS)
            volume_confirmed = vol_ratio >= VOLUME_MULT
            
            if stoch_cross_down and volume_confirmed:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals