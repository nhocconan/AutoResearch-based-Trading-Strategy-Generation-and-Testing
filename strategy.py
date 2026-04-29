#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R Mean Reversion + 1w Supertrend trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe for mean reversion entries.
# Weekly Supertrend filters for higher timeframe trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakout/mean reversion has participation.
# ATR-based trailing stop manages risk while allowing trends to run.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_WilliamsR_MeanRev_1wSupertrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter
    hl2_1w = (df_1w['high'].values + df_1w['low'].values) / 2
    atr_1w = pd.Series(np.maximum.reduce([
        df_1w['high'].values[1:] - df_1w['low'].values[1:],
        np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1]),
        np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    ])).rolling(window=10, min_periods=10).mean().values
    atr_1w = np.concatenate([[np.nan], atr_1w])
    
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    supertrend_1w = np.full_like(hl2_1w, np.nan, dtype=float)
    direction_1w = np.full_like(hl2_1w, np.nan, dtype=float)
    
    for i in range(1, len(hl2_1w)):
        if np.isnan(upper_band_1w[i]) or np.isnan(lower_band_1w[i]) or np.isnan(close_1w := df_1w['close'].values[i]):
            continue
        if i == 1:
            supertrend_1w[i] = lower_band_1w[i]
            direction_1w[i] = 1
        else:
            if supertrend_1w[i-1] == upper_band_1w[i-1]:
                supertrend_1w[i] = lower_band_1w[i] if close_1w <= upper_band_1w[i] else upper_band_1w[i]
                direction_1w[i] = 1 if supertrend_1w[i] == lower_band_1w[i] else -1
            else:
                supertrend_1w[i] = upper_band_1w[i] if close_1w >= lower_band_1w[i] else lower_band_1w[i]
                direction_1w[i] = -1 if supertrend_1w[i] == upper_band_1w[i] else 1
    
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Calculate Williams %R(14) for mean reversion signals
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime filter (avoid high volatility chop)
    atr_percentile = pd.Series(atr).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 20 else np.nan, raw=True
    ).values
    vol_regime_filter = atr <= atr_percentile  # Only trade in low/medium volatility regimes
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    
    start_idx = max(50, 20, 14)  # warmup for Supertrend, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(direction_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_direction_1w = direction_1w_aligned[i]
        curr_williams_r = williams_r[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_regime = vol_regime_filter[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Williams %R crosses above -20 (overbought - mean reversion exit)
            # 3. Weekly trend turns bearish
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_low <= stop_price or
                curr_williams_r > -20 or
                curr_direction_1w < 0 or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Williams %R crosses below -80 (oversold - mean reversion exit)
            # 3. Weekly trend turns bullish
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_high >= stop_price or
                curr_williams_r < -80 or
                curr_direction_1w > 0 or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter in low/medium volatility regimes to avoid whipsaws
            if not curr_vol_regime:
                signals[i] = 0.0
                continue
                
            # Long entry: Williams %R crosses above -80 (oversold recovery) + weekly bullish trend + volume confirm
            if (curr_williams_r > -80 and
                curr_direction_1w > 0 and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: Williams %R crosses below -20 (overbought decline) + weekly bearish trend + volume confirm
            elif (curr_williams_r < -20 and
                  curr_direction_1w < 0 and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals