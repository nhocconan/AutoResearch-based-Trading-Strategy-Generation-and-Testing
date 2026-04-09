#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility regime
# - Uses 1h RSI(14) for mean reversion signals (long when RSI<30, short when RSI>70)
# - Filters by 4h EMA(50) trend: only long when price>EMA50, short when price<EMA50
# - Volatility regime filter: trade only when 1d ATR(14) > 20-period median ATR (high vol environments)
# - Exits on RSI mean reversion (RSI>50 for longs, RSI<50 for shorts) or opposite EMA cross
# - Position size: 0.20 (20% of capital) to limit drawdown in 2022-like crashes
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)
# - Combines mean reversion timing with trend and volatility filters to avoid false signals

name = "1h_4h_1d_rsi_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # 1d ATR(20) for median calculation
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    # Median of ATR(20) over 50 periods for regime filter
    median_atr_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).median().values
    # High volatility regime: current ATR > median ATR
    vol_regime = atr_20 > median_atr_50
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # 1h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_regime_aligned[i]) or
            np.isnan(rsi[i]) or vol_regime_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: RSI mean reversion or price below EMA
            if rsi[i] >= 50 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: RSI mean reversion or price above EMA
            if rsi[i] <= 50 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for RSI extreme with trend and volatility filters
            if (rsi[i] <= 30 and  # Oversold
                close[i] > ema_4h_aligned[i] and  # Uptrend filter
                vol_regime_aligned[i]):  # High volatility regime
                position = 1
                signals[i] = 0.20
            elif (rsi[i] >= 70 and  # Overbought
                  close[i] < ema_4h_aligned[i] and  # Downtrend filter
                  vol_regime_aligned[i]):  # High volatility regime
                position = -1
                signals[i] = -0.20
    
    return signals