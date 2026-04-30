#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1/S1) with 1d EMA34 trend filter and volume confirmation
# Uses 1d HTF for Camarilla pivot calculation (key intraday support/resistance) and EMA trend to avoid whipsaws.
# Long when price breaks above R1 in uptrend (close > EMA34) with volume confirmation.
# Short when price breaks below S1 in downtrend (close < EMA34) with volume confirmation.
# Designed for low trade frequency (~15-25/year on 12h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at extreme levels.
# Focus on BTC/ETH as primary targets.

name = "12h_1dCamarilla_R1S1_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1) using typical price
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tp_1d = typical_price.values
    
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 50-period average (moderate to balance trades and confirmation)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_confirm = volume[i] > (1.5 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation and trend alignment
            if volume_confirm:
                # Bullish entry: price breaks above 1d R1 with 1d uptrend (close > EMA34)
                if curr_close > curr_r1 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d S1 with 1d downtrend (close < EMA34)
                elif curr_close < curr_s1 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: price breaks below 1d S1 (reversal signal)
            if curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d R1 (mean reversion at extreme level)
            elif curr_close > curr_r1:
                signals[i] = 0.0  # exit at opposite level
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price breaks above 1d R1 (reversal signal)
            if curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d S1 (mean reversion at extreme level)
            elif curr_close < curr_s1:
                signals[i] = 0.0  # exit at opposite level
            else:
                signals[i] = -0.25
    
    return signals