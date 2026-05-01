#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend + volume spike.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume > 2.0x 20-period 6h volume average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume > 2.0x 20-period 6h volume average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Alligator calculated from SMAs: jaws=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3) on median price.
# Works in bull (Alligator alignment + uptrend) and bear (Alligator alignment + downtrend).
# Target: 12-37 trades/year on 6h timeframe.

name = "6h_WilliamsAlligator_1dEMA34_Volume_v2"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator on 6h data
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    # Jaws: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume average (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Alligator (max shift 8) and ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaws.iloc[i]) or 
            np.isnan(teeth.iloc[i]) or 
            np.isnan(lips.iloc[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment
        jaw_val = jaws.iloc[i]
        tooth_val = teeth.iloc[i]
        lip_val = lips.iloc[i]
        bullish_align = jaw_val < tooth_val < lip_val
        bearish_align = jaw_val > tooth_val > lip_val
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 6h volume average
        if vol_ma_6h[i] <= 0 or np.isnan(vol_ma_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_6h[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment AND uptrend AND volume confirmation
            if bullish_align and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator alignment AND downtrend AND volume confirmation
            elif bearish_align and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks OR trend reverses
            elif not bullish_align or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks OR trend reverses
            elif not bearish_align or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals