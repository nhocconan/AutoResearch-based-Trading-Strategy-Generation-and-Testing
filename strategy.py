#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses ATR-based dynamic position sizing (0.20-0.30) to control drawdown in bear markets
# Trend filter (1d EMA34) allows long in bull/uptrend, short in bear/downtrend
# Volume confirmation ensures breakout strength
# Target: 20-40 trades/year (80-160 total) to minimize fee drag
# Works in both bull and bear via trend filter - only takes trades in direction of 1d trend

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_DynamicSize"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility measurement and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime (avoid extreme volatility)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=20).mean().values
    atr_std_50 = pd.Series(atr).rolling(window=50, min_periods=20).std().values
    atr_upper = atr_ma_50 + 2.0 * atr_std_50
    vol_regime_filter = atr <= atr_upper  # Avoid extreme volatility outliers
    
    # Calculate Camarilla levels from previous day
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    # Align previous day's data to 4h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_aligned + 1.0 * (prev_high_aligned - prev_low_aligned)
    camarilla_s3 = prev_close_aligned - 1.0 * (prev_high_aligned - prev_low_aligned)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 34, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_regime = vol_regime_filter[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below 1d EMA34 (trend change)
            # 2. Price drops below Camarilla S3 (breakout failed)
            # 3. Volatility regime shifts to extreme (avoid chop)
            if (curr_close < curr_ema_34_1d or
                curr_close < curr_s3 or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                # Dynamic position sizing: larger size in lower volatility
                vol_factor = np.clip(atr_ma_50[i] / (curr_atr + 1e-10), 0.5, 1.5)
                base_size = 0.25
                signals[i] = base_size * vol_factor
                # Clamp to max 0.35
                if signals[i] > 0.35:
                    signals[i] = 0.35
                    
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above 1d EMA34 (trend change)
            # 2. Price rises above Camarilla R3 (breakout failed)
            # 3. Volatility regime shifts to extreme (avoid chop)
            if (curr_close > curr_ema_34_1d or
                curr_close > curr_r3 or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                # Dynamic position sizing: larger size in lower volatility
                vol_factor = np.clip(atr_ma_50[i] / (curr_atr + 1e-10), 0.5, 1.5)
                base_size = 0.25
                signals[i] = -base_size * vol_factor
                # Clamp to min -0.35
                if signals[i] < -0.35:
                    signals[i] = -0.35
                    
        else:  # Flat - look for new entries
            # Only enter in non-extreme volatility regimes
            if not curr_vol_regime:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Camarilla R3 + above 1d EMA34 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: price breaks below Camarilla S3 + below 1d EMA34 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals