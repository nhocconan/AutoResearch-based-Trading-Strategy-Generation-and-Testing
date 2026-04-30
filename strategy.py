#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels (R1/S1) with 4h EMA21 trend filter and volume spike confirmation
# Uses 4h HTF for Camarilla pivot calculation and EMA trend to avoid whipsaws.
# Long when price breaks above R1 in uptrend (close > EMA21) with volume spike during active session (08-20 UTC).
# Short when price breaks below S1 in downtrend (close < EMA21) with volume spike during active session.
# Designed for low trade frequency (~15-35/year on 1h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at extreme levels.
# Focus on BTC/ETH as primary targets.

name = "1h_4hCamarilla_R1S1_Breakout_4hEMA21_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla and EMA calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1) using typical price
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tp_4h = typical_price.values
    
    # Camarilla: R1 = close + 1.1*(high-low)/4, S1 = close - 1.1*(high-low)/4
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 4
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 4
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate ATR(14) for dynamic stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA(21)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0x 50-period average (strict to reduce trades)
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
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema = ema_21_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike, session, and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h R1 with 4h uptrend (close > EMA21)
                if curr_close > curr_r1 and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h S1 with 4h downtrend (close < EMA21)
                elif curr_close < curr_s1 and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h S1 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry OR touches 4h R1 (mean reversion)
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h R1 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry OR touches 4h S1 (mean reversion)
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.20
    
    return signals