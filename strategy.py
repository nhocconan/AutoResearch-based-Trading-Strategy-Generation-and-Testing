#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR regime filter and volume confirmation.
# Uses 1w ATR to detect high/low volatility regimes - breakouts work best in expanding volatility.
# Long when: price breaks above Donchian(20) high AND 1w ATR > 20-period mean AND volume > 1.5x average.
# Short when: price breaks below Donchian(20) low AND 1w ATR > 20-period mean AND volume > 1.5x average.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 15-25 trades/year.
# Donchian channels provide clear structure, ATR regime filter avoids false breakouts in low volatility,
# volume confirmation ensures institutional participation. Works in bull (breakouts up) and bear (breakouts down).

name = "1d_Donchian20_1wATR_Regime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency (optional for 1d but kept for consistency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ATR(20) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan  # First bar has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20) - using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1w = np.full_like(tr, np.nan)
    if len(tr) >= 20:
        atr_1w[19] = np.nanmean(tr[:20])  # First ATR is simple average
        for i in range(20, len(tr)):
            if not np.isnan(atr_1w[i-1]):
                atr_1w[i] = (atr_1w[i-1] * 19 + tr[i]) / 20
            else:
                atr_1w[i] = np.nan
    
    # Calculate 1w ATR(20) mean for regime threshold
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w ATR and ATR mean to 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Calculate Donchian(20) on 1d
    # Donchian high = highest high over last 20 periods
    # Donchian low = lowest low over last 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Optional session filter: 00-24 UTC (always active for 1d, but keeping structure)
        hour = hours[i]
        in_session = True  # 1d timeframe always in session
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_atr = atr_1w_aligned[i]
        curr_atr_ma = atr_ma_1w_aligned[i]
        curr_vol_ma = vol_ma[i]
        
        # Regime filter: only trade when ATR is above its mean (expanding volatility)
        vol_regime = curr_atr > curr_atr_ma
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = curr_volume > (1.5 * curr_vol_ma)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND vol regime AND vol confirm
            if (curr_close > curr_donch_high and 
                vol_regime and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND vol regime AND vol confirm
            elif (curr_close < curr_donch_low and 
                  vol_regime and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: break below Donchian low (opposite side) OR volatility contracts
            if (curr_close < curr_donch_low or 
                not vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: break above Donchian high (opposite side) OR volatility contracts
            if (curr_close > curr_donch_high or 
                not vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals