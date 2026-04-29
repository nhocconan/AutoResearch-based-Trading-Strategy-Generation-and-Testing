#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe to capture medium-term swings while avoiding excessive trading.
# Camarilla levels from 1d provide key support/resistance; 1d EMA34 filters for trend alignment;
# volume confirms breakout strength. Designed to work in both bull and bear markets by
# following the higher timeframe trend while using volatility-adjusted position sizing.
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility-based position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime filter
    atr_percentile = pd.Series(atr).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 20 else np.nan, raw=True
    ).values
    vol_regime_filter = ~np.isnan(atr_percentile) & (atr <= atr_percentile * 1.5)  # Avoid extreme volatility
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    # Camarilla R3 and S3 levels (most significant levels for breakout)
    camarilla_r3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.8x 30-period average to avoid false breakouts
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30, 14)  # warmup for EMA34, volume MA, ATR
    
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
            # 1. Price drops below Camarilla S3 (breakout failed)
            # 2. Price crosses below 1d EMA34 (trend change)
            # 3. Volume confirmation lost (weakening momentum)
            if (curr_close < curr_s3 or
                curr_close < curr_ema_34_1d or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                # Scale position based on volatility (inverse ATR)
                vol_scaling = np.clip(0.5 / (curr_atr / curr_close + 0.01), 0.5, 1.0)
                signals[i] = 0.25 * vol_scaling
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Camarilla R3 (breakout failed)
            # 2. Price crosses above 1d EMA34 (trend change)
            # 3. Volume confirmation lost (weakening momentum)
            if (curr_close > curr_r3 or
                curr_close > curr_ema_34_1d or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                # Scale position based on volatility (inverse ATR)
                vol_scaling = np.clip(0.5 / (curr_atr / curr_close + 0.01), 0.5, 1.0)
                signals[i] = -0.25 * vol_scaling
                
        else:  # Flat - look for new entries
            # Only enter in favorable volatility regimes to avoid whipsaws
            if not curr_vol_regime:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Camarilla R3 + above 1d EMA34 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 + below 1d EMA34 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals