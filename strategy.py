#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness Index regime filter.
# Uses 1d timeframe for signal generation.
# KAMA(10,2,30) for adaptive trend direction to avoid whipsaws in ranging markets.
# RSI(14) for momentum confirmation (long when RSI>50, short when RSI<50).
# Choppiness Index(14) > 61.8 for ranging regime (mean reversion at Bollinger Bands),
# < 38.2 for trending regime (trend following with KAMA).
# Volume confirmation: current volume > 1.3x 20-bar average.
# Discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag.

name = "1d_KAMA_RSI_Chop_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA(10,2,30) for trend direction
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for momentum
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index(14) for regime filter
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 30  # warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(kama[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.3)
        
        # Regime filter based on Choppiness Index
        chop_value = chop[i]
        ranging_regime = chop_value > 61.8
        trending_regime = chop_value < 38.2
        
        # KAMA direction: price above KAMA = uptrend, below = downtrend
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # RSI momentum: >50 bullish, <50 bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA uptrend AND RSI bullish AND volume confirmation
            # In trending regime: follow KAMA+RSI
            # In ranging regime: mean reversion at extremes (RSI<30 for long, RSI>70 for short)
            if (uptrend and 
                rsi_bullish and 
                volume_confirm and
                trending_regime):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA downtrend AND RSI bearish AND volume confirmation
            elif (downtrend and 
                  rsi_bearish and 
                  volume_confirm and
                  trending_regime):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            # Ranging regime: mean reversion at Bollinger Band-like extremes
            elif (ranging_regime and
                  volume_confirm):
                # Long when oversold (RSI<30) and price near recent low
                if (rsi[i] < 30 and 
                    curr_close <= ll[i] * 1.02):  # within 2% of recent low
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short when overbought (RSI>70) and price near recent high
                elif (rsi[i] > 70 and 
                      curr_close >= hh[i] * 0.98):  # within 2% of recent high
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
            # Exit: trend changes to downtrend OR RSI becomes bearish
            elif not uptrend or not rsi_bullish:
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
            # Exit: trend changes to uptrend OR RSI becomes bullish
            elif not downtrend or not rsi_bearish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals