#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels (R4/S4) with 12h EMA50 trend filter and volume spike confirmation
# Uses 12h HTF for Camarilla pivot calculation to reduce noise and 12h EMA50 for trend to filter false breakouts.
# Long when price breaks above 12h R4 in uptrend (4h close > 12h EMA50) with volume spike (>2.0x average).
# Short when price breaks below 12h S4 in downtrend (4h close < 12h EMA50) with volume spike.
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>2.0x average) to balance signal quality and trade count.
# Stoploss at 2.5 * ATR and take profit at 2.0 * ATR to allow for larger swings on 4h timeframe.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 12h pivot levels.
# Focus on BTC/ETH as primary targets.

name = "4h_12hCamarilla_R4S4_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R4, S4) using typical price
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    camarilla_r4 = close_12h + 1.1 * (high_12h - low_12h)
    camarilla_s4 = close_12h - 1.1 * (high_12h - low_12h)
    
    # Align 12h Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average (moderate to balance trades)
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
                # Bullish entry: price breaks above 12h R4 with 12h uptrend (close > EMA50)
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 12h S4 with 12h downtrend (close < EMA50)
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks 12h S4 (reversal signal)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR above entry
            elif curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks 12h R4 (reversal signal)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR below entry
            elif curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals