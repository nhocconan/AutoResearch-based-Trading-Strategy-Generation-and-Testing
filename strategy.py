#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla R3/S3 breakout with 1d EMA(34) trend filter, volume confirmation, and chop regime filter
# Uses 1d HTF for institutional Camarilla pivot levels and EMA trend, with choppiness index to avoid whipsaws in ranging markets.
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion at extreme levels.
# Focus on BTC/ETH as primary targets.

name = "12h_1dCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar's OHLC)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = typical_price + hl_range * 1.1 / 4
    camarilla_s3 = typical_price - hl_range * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(atr14) / (max(high,n) - min(low,n))) / log10(n)
    # We use 14-period CHOP on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1_1d[0], tr2_1d[0], tr3_1d[0]])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: avoid division by zero
    range_14 = max_high_14 - min_low_14
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral if undefined
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34) and CHOP
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_chop = chop_aligned[i]
        
        # Regime filter: only trade when CHOP < 61.8 (trending market)
        in_trending_regime = curr_chop < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike, trend alignment, and trending regime
            if volume_spike and in_trending_regime:
                # Bullish entry: price breaks above 1d R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d R4 (mean reversion tendency)
            # R4 = C + (H-L)*1.1/2 = R3 + (H-L)*1.1/4
            hl_range_1d = (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_r4 = typical_price_1d + hl_range_1d * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_r4))[i] if len(df_1d) > 0 else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d S4 (mean reversion tendency)
            # S4 = C - (H-L)*1.1/2 = S3 - (H-L)*1.1/4
            hl_range_1d = (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_s4 = typical_price_1d - hl_range_1d * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_s4))[i] if len(df_1d) > 0 else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals