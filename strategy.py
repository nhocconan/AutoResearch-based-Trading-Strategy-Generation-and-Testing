#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels (R4/S4) with 1d EMA50 trend filter and volume spike confirmation
# Uses 1w HTF for Camarilla pivot calculation (R4/S4) for strong breakout signals and 1d EMA50 for trend to filter false breakouts.
# Long when price breaks above 1w R4 in uptrend (1d close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 1w S4 in downtrend (1d close < 1d EMA50) with volume spike.
# Designed for low trade frequency (~7-25/year on 1d) to minimize fee drag while capturing strong directional moves.
# Uses moderate volume confirmation (>2.0x average) and outer Camarilla levels (R4/S4) to balance signal quality and frequency.
# Stoploss at 2.0 * ATR and take profit at 3.0 * ATR for asymmetric risk-reward (1:1.5).
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 1w pivot levels.
# Focus on BTC/ETH as primary targets.

name = "1d_1wCamarilla_R4S4_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (R4, S4) using typical price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: R4 = close + 1.1*(high-low)/2, S4 = close - 1.1*(high-low)/2
    camarilla_r4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_s4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align 1w Camarilla levels to 1d timeframe (wait for 1w bar to close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for dynamic stoploss on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1w R4 with 1d uptrend (close > EMA50)
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1w S4 with 1d downtrend (close < EMA50)
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1w S4 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR above entry
            elif curr_close > entry_price + 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1w R4 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR below entry
            elif curr_close < entry_price - 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals