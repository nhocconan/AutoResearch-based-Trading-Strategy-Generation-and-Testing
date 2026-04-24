#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA34 for trend filter (price above/below EMA34 defines bull/bear regime).
- Entry: Long when price breaks above Donchian H20 in bull regime with volume > 2.0 * 1d volume MA(20);
         Short when price breaks below Donchian L20 in bear regime with volume > 2.0 * 1d volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian H20/L20 are strong structural levels, reducing false breakouts.
- Works in bull (breakouts with trend) and bear (strong moves after panic lows/highs).
- Uses actual 1w data from mtf_data to avoid look-ahead and resampling issues.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian levels (H20, L20) on 1d data using previous 20 periods
    # Donchian H20 = max(high of last 20 periods), L20 = min(low of last 20 periods)
    # Using rolling window on 1d data, then align to 1d timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_h20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_l20 = low_series.rolling(window=20, min_periods=20).min().values
    # Shift to avoid look-ahead: levels calculated from current bar apply to next bar
    donchian_h20 = np.roll(donchian_h20, 1)
    donchian_l20 = np.roll(donchian_l20, 1)
    donchian_h20[0] = np.nan
    donchian_l20[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14, 1)  # EMA34 needs 34, Donchian needs 20, ATR needs 14, plus 1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(donchian_h20[i]) or np.isnan(donchian_l20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 2.0x threshold (tight to reduce trades)
        vol_spike = curr_volume > 2.0 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA34
        bull_regime = curr_close > ema_34_1w_aligned[i]
        bear_regime = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian H20 in bull regime with volume spike
            if curr_close > donchian_h20[i] and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian L20 in bear regime with volume spike
            elif curr_close < donchian_l20[i] and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below L20)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or curr_close < donchian_l20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above H20)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or curr_close > donchian_h20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0