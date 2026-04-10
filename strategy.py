#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI with 1w trend filter and 1d volatility regime
# - VW-RSI(14): RSI calculated using typical price * volume as input, gives more weight to high-volume moves
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - 1d ATR percentile regime filter: only trade when ATR(14) is between 30th and 70th percentile (avoid extremes)
# - Entry logic:
#   * Long: VW-RSI < 30 AND price > weekly EMA50 AND ATR in mid-range
#   * Short: VW-RSI > 70 AND price < weekly EMA50 AND ATR in mid-range
# - Exit: opposite VW-RSI signal (Long exit at VW-RSI > 50, Short exit at VW-RSI < 50)
# - Discrete position sizing (0.25) to minimize fee churn
# - VW-RSI reduces false signals in low-volume moves and captures institutional participation
# - ATR regime filter avoids choppy (low ATR) and volatile (high ATR) extremes
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_vwrsi_trend_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d ATR for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 1d ATR percentile rank (30th and 70th) for regime filter
    # We'll calculate rolling percentile using min_periods
    atr_series = pd.Series(atr_1d)
    atr_pct_30 = atr_series.rolling(window=50, min_periods=30).quantile(0.30).values
    atr_pct_70 = atr_series.rolling(window=50, min_periods=30).quantile(0.70).values
    atr_in_mid_range = (atr_1d >= atr_pct_30) & (atr_1d <= atr_pct_70)
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_in_mid_range_aligned = align_htf_to_ltf(prices, df_1d, atr_in_mid_range.astype(float))
    
    # Pre-compute 6h typical price and volume for VW-RSI
    typical_price = (prices['high'].values + prices['low'].values + prices['close'].values) / 3.0
    volume = prices['volume'].values
    vp = typical_price * volume  # volume-weighted price
    
    # Calculate VW-RSI(14) on 6h timeframe
    change = vp - np.roll(vp, 1)
    change[0] = 0
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, -change, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    loss_smooth = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = gain_smooth / (loss_smooth + 1e-10)  # avoid division by zero
    vwrsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vwrsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_in_mid_range_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        vwrsi_current = vwrsi[i]
        ema_50_current = ema_50_aligned[i]
        atr_regime_current = bool(atr_in_mid_range_aligned[i])  # Convert from float to bool
        close_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: VW-RSI oversold (<30), price above weekly EMA, ATR in mid-range
            if (vwrsi_current < 30 and 
                close_price > ema_50_current and 
                atr_regime_current):
                position = 1
                signals[i] = 0.25
            # Short conditions: VW-RSI overbought (>70), price below weekly EMA, ATR in mid-range
            elif (vwrsi_current > 70 and 
                  close_price < ema_50_current and 
                  atr_regime_current):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: VW-RSI crosses midline (50) in opposite direction
            if position == 1:  # Long position
                if vwrsi_current > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                if vwrsi_current < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals