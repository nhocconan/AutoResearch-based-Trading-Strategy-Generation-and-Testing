#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels (R3/S3) with 1d EMA34 trend filter and volume spike confirmation
# Uses 1w HTF for Camarilla pivot calculation (weekly support/resistance) and 1d EMA for trend filter.
# Long when price breaks above 1w R3 in uptrend (close > EMA34) with volume spike (>2.0x average).
# Short when price breaks below 1w S3 in downtrend (close < EMA34) with volume spike.
# Designed for low trade frequency (~15-25/year on 1d) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>2.0x average) to balance signal quality and trade frequency.
# Stoploss at 2.0 * ATR and take profit at 1.5 * ATR to limit losing trades and secure gains.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at pivot levels.
# Focus on BTC/ETH as primary targets.

name = "1d_1wCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter (though prices is already 1d, we use it directly)
    # Calculate 1w Camarilla levels (R3, S3) using typical price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align 1w Camarilla levels to 1d timeframe (wait for 1w bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate EMA(34) for trend filter on 1d timeframe
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for dynamic stoploss on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average (moderate to balance trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema = ema_34[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1w R3 with uptrend (close > EMA34)
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1w S3 with downtrend (close < EMA34)
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1w S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1w R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals