#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime + volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# In trending markets (1d ADX>=25): long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
# In ranging markets (1d ADX<25): fade extremes using 6h RSI(14) <30/>70 with volume confirmation.
# Volume confirmation (>1.5x 20-period EMA) reduces false signals. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_ElderRay_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h RSI(14) for ranging market mean reversion
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: mean reversion using RSI extremes
                if rsi_values[i] < 30 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif rsi_values[i] > 70 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: Elder Ray trend following
                # Long: Bull Power > 0 and rising (previous bull power < current)
                if (bull_power[i] > 0 and 
                    i > 100 and bull_power[i] > bull_power[i-1] and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 and rising (previous bear power < current)
                elif (bear_power[i] > 0 and 
                      i > 100 and bear_power[i] > bear_power[i-1] and 
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Elder Ray weakening OR RSI overbought OR volume drops
            if (bull_power[i] <= 0 or 
                rsi_values[i] > 70 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray weakening OR RSI oversold OR volume drops
            if (bear_power[i] <= 0 or 
                rsi_values[i] < 30 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals