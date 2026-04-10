#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume confirmation and ADX trend filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.3x 20-period volume SMA AND 1d ADX > 25 (trending market)
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.3x 20-period volume SMA AND 1d ADX > 25 (trending market)
# - Exit: Williams %R crosses below -50 for longs or above -50 for shorts, or ATR trailing stop (2.0 * ATR)
# - Uses 4h for Williams %R and price action, 1d for volume and ADX confirmation
# - Williams %R captures short-term reversals in trending markets
# - Volume spike adds conviction to reversal signals
# - ADX filter ensures we trade only when there is sufficient trend strength
# - ATR trailing stop manages risk while allowing profits to run
# - Target: 25-40 trades/year to balance opportunity with fee drag

name = "4h_1d_williamsr_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and ADX confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute Williams %R for 4h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute ATR for trailing stop (using 4h data)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(14, n):  # Start after 14-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA (volume spike)
        vol_confirm = vol_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d ADX > 25 (sufficient trend strength)
        trend_filter = adx_aligned[i] > 25
        
        # Williams %R signals
        wr_long_signal = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80
        wr_short_signal = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20
        wr_exit_long = williams_r[i] < -50  # Cross below -50 for long exit
        wr_exit_short = williams_r[i] > -50  # Cross above -50 for short exit
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Update trailing stop extremes
            if position == 1:
                # Long position: track highest high for trailing stop
                pass  # Will use Williams %R for exit primarily
            elif position == -1:
                # Short position: track lowest low for trailing stop
                pass  # Will use Williams %R for exit primarily
            
            # Long: Williams %R crosses above -80 (oversold reversal)
            if wr_long_signal:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Williams %R crosses below -20 (overbought reversal)
            elif wr_short_signal:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position or check for exits
                if position == 1 and wr_exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and wr_exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Additional exit: ATR trailing stop (2.0 * ATR from entry)
            # Simplified: use Williams %R as primary exit, ATR as backup
            if position == 1 and i >= 2:
                # Check if price has moved against us significantly
                if close[i] < close[i-1] - 2.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1 and i >= 2:
                if close[i] > close[i-1] + 2.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
        else:
            # No volume or trend confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals