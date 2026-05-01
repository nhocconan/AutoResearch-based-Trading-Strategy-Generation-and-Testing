#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs. 
# Alligator sleeping (jaw-teeth-lips intertwined) = range market -> fade at R3/S3.
# Alligator awakening (jaw-teeth-lips separated, aligned) = trending -> breakout continuation.
# Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13.
# Long: Alligator awakening + Bull Power > 0 + price > Camarilla R3.
# Short: Alligator awakening + Bear Power < 0 + price < Camarilla S3.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Combines trend (Alligator), momentum (Elder Ray), and structure (Camarilla) for BTC/ETH in bull/bear.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_WilliamsAlligator_ElderRay_Camarilla_v1"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels (using prior bar's OHLC to avoid look-ahead)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + 1.125 * camarilla_range
    camarilla_S3 = prev_close - 1.125 * camarilla_range
    
    # Calculate 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Williams Alligator conditions
        # Alligator sleeping: jaw, teeth, lips intertwined (market ranging)
        alligator_sleeping = (abs(jaw[i] - teeth[i]) < (atr[i] * 0.1) and 
                              abs(teeth[i] - lips[i]) < (atr[i] * 0.1) and
                              abs(lips[i] - jaw[i]) < (atr[i] * 0.1))
        # Alligator awakening: jaw, teeth, lips separated and aligned (trending)
        alligator_awakening_up = (jaw[i] > teeth[i] > lips[i])  # aligned up
        alligator_awakening_down = (jaw[i] < teeth[i] < lips[i])  # aligned down
        
        if position == 0:  # Flat - look for new entries
            # In ranging market (alligator sleeping): fade at Camarilla R3/S3
            if alligator_sleeping:
                # Long: price < Camarilla S3 AND bull power positive AND volume spike
                if curr_close < camarilla_S3[i] and bull_power[i] > 0 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price > Camarilla R3 AND bear power negative AND volume spike
                elif curr_close > camarilla_R3[i] and bear_power[i] < 0 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                else:
                    signals[i] = 0.0
            # In trending market (alligator awakening): breakout continuation
            else:
                # Long: price > Camarilla R3 AND alligator awakening up AND uptrend AND volume spike
                if curr_close > camarilla_R3[i] and alligator_awakening_up and uptrend and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price < Camarilla S3 AND alligator awakening down AND downtrend AND volume spike
                elif curr_close < camarilla_S3[i] and alligator_awakening_down and downtrend and volume_confirm:
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
            # Exit: price breaks below Camarilla S3 OR alligator turns sleeping OR trend turns down
            elif (curr_close < camarilla_S3[i] or 
                  alligator_sleeping or 
                  not uptrend):
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
            # Exit: price breaks above Camarilla R3 OR alligator turns sleeping OR trend turns up
            elif (curr_close > camarilla_R3[i] or 
                  alligator_sleeping or 
                  not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals