#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA34 trend filter and volume confirmation
# Williams %R > -20 = overbought, < -80 = oversold. Mean reversion in range markets.
# In strong trends (price > EMA34 for long, price < EMA34 for short), we fade extreme %R only when aligned with trend.
# Uses volume spike (>2.0x average) to confirm participation. Designed for low trade frequency (~12-37/year on 6h).
# Works in bull markets via buying oversold pullbacks in uptrend, in bear markets via selling overbought bounces in downtrend.

name = "6h_1dWilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Align 1d Williams %R to 6h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for dynamic stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # warmup for ATR and volume average
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 40-period average
        if i >= 40:
            vol_ma_40 = np.mean(volume[i-40:i])
        elif i > 0:
            vol_ma_40 = np.mean(volume[:i])
        else:
            vol_ma_40 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_40) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_williams = williams_r_aligned[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Long: Williams %R < -80 (oversold) and price > EMA34 (uptrend)
                if curr_williams < -80 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Williams %R > -20 (overbought) and price < EMA34 (downtrend)
                elif curr_williams > -20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Williams %R > -20 (overbought exit)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams > -20:
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R > -50 (momentum exit)
            elif curr_williams > -50:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Williams %R < -80 (oversold exit)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams < -80:
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R < -50 (momentum exit)
            elif curr_williams < -50:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals