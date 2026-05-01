#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 (uptrend) AND volume > 2.0x 20-period median.
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 (downtrend) AND volume > 2.0x 20-period median.
# Exit when price returns to Camarilla H3/L3 level or trend reverses.
# Camarilla levels provide precise intraday support/resistance; EMA34 filters counter-trend trades; volume confirms breakout strength.
# Target: 15-35 trades/year on 6h timeframe. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous 6h bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # We use previous bar's high/low/close to avoid look-ahead
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + rang * 1.1 * 1.1 / 4
    camarilla_s3 = prev_close - rang * 1.1 * 1.1 / 4
    camarilla_h3 = prev_close + rang * 1.1 / 2
    camarilla_l3 = prev_close - rang * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for EMA34, ATR, and volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_long = curr_close > camarilla_r3[i]
        breakout_short = curr_close < camarilla_s3[i]
        return_to_h3 = curr_close < camarilla_h3[i]
        return_to_l3 = curr_close > camarilla_l3[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakdown below S3 AND downtrend AND volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: Return to H3 OR trend reversal
            if return_to_h3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Return to L3 OR trend reversal
            if return_to_l3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals