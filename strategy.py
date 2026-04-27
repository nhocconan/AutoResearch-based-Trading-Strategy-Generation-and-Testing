# Inverted-VoR-12h
# Hypothesis: 12h mean-reversion using intraday volatility regime (ATR ratio) and RSI extremes.
# Works in bull/bear because it fades extremes when volatility is elevated, which occurs in both regimes.
# Uses 1d trend filter (EMA34) to align with higher timeframe direction and avoid counter-trend whipsaws.
# Entry: RSI < 30 (long) or > 70 (short) + ATR(12)/ATR(48) > 1.5 (volatility spike) + price vs 1d EMA34.
# Exit: RSI crosses back to neutral (40/60) or volatility contraction (ATR ratio < 1.2).
# Position size: 0.25 to limit drawdown. Expected trades: ~25-40/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 48:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(48) for volatility regime (using daily high/low/close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_48_1d = pd.Series(tr).ewm(span=48, adjust=False, min_periods=48).mean().values
    atr_48_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_48_1d)
    
    # Calculate 12-period ATR for current timeframe
    tr_l = high - low
    tr_l2 = np.abs(high - np.roll(close, 1))
    tr_l3 = np.abs(low - np.roll(close, 1))
    tr_l2[0] = tr_l[0]
    tr_l3[0] = tr_l[0]
    tr_l = np.maximum(tr_l, np.maximum(tr_l2, tr_l3))
    atr_12 = pd.Series(tr_l).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 12-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility regime filter: current 12-period ATR vs daily 48-period ATR
    vol_regime = atr_12 / (atr_48_1d_aligned + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_48_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Mean reversion signals: RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_exit_long = rsi[i] > 40
        rsi_exit_short = rsi[i] < 60
        
        # Volatility filter: elevated volatility (regime > 1.5)
        vol_filter = vol_regime[i] > 1.5
        
        # Long conditions: oversold RSI + volatility spike + bullish trend bias
        long_condition = (rsi_oversold and 
                         vol_filter and 
                         price_above_ema)
        
        # Short conditions: overbought RSI + volatility spike + bearish trend bias
        short_condition = (rsi_overbought and 
                          vol_filter and 
                          price_below_ema)
        
        # Exit conditions: RSI mean reversion or volatility contraction
        exit_long = (position == 1 and (rsi_exit_long or vol_regime[i] < 1.2))
        exit_short = (position == -1 and (rsi_exit_short or vol_regime[i] < 1.2))
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "Inverted-VoR-12h"
timeframe = "12h"
leverage = 1.0