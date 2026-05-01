#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND close > 1w EMA34 AND volume > 1.5x 20-day volume median.
# Short when price breaks below Camarilla S3 AND close < 1w EMA34 AND volume > 1.5x 20-day volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 20-40 trades/year on 1d timeframe (~80-160 total over 4 years).
# Proven pattern: Camarilla breakouts with volume and trend filter work on BTC/ETH in both bull/bear markets.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day volume median for volume confirmation
    vol_median_20d = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels (R3, S3) from prior day to avoid look-ahead
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width = (high - low) * 1.1 / 12.0
    # R3 = close + camarilla_width * 1.1
    # S3 = close - camarilla_width * 1.1
    camarilla_r3 = close + camarilla_width * 1.1
    camarilla_s3 = close - camarilla_width * 1.1
    # Shift by 1 to use prior day's levels
    camarilla_r3_prior = np.roll(camarilla_r3, 1)
    camarilla_s3_prior = np.roll(camarilla_s3, 1)
    camarilla_r3_prior[0] = np.nan
    camarilla_s3_prior[0] = np.nan
    
    # Calculate 1w EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_prior[i]) or 
            np.isnan(camarilla_s3_prior[i]) or 
            np.isnan(vol_median_20d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day volume median
        if vol_median_20d[i] <= 0 or np.isnan(vol_median_20d[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20d[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Camarilla R3 AND uptrend AND volume spike
            if curr_close > camarilla_r3_prior[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Camarilla S3 AND downtrend AND volume spike
            elif curr_close < camarilla_s3_prior[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Camarilla S3 OR trend turns down
            elif curr_close < camarilla_s3_prior[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Camarilla R3 OR trend turns up
            elif curr_close > camarilla_r3_prior[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals