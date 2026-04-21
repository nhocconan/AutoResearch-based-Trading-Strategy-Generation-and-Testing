#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) with extreme readings (oversold <25, overbought >75) 
# combined with weekly trend filter (weekly EMA50) and volume confirmation.
# Works in bull/bear: In bear market, oversold bounces during weekly uptrend; 
# in bull market, overbought reversals during weekly downtrend. 
# Weekly trend ensures we trade with higher timeframe momentum.
# Low frequency (~10-20 trades/year) minimizes fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily RSI(14) for mean reversion signals ===
    close_1d = prices['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_trend = ema_50_1w_aligned[i]
        rsi_val = rsi_values[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: RSI oversold + weekly uptrend + volume confirmation
            if (rsi_val < 25 and  # Oversold
                price_close > weekly_trend and  # Above weekly EMA50 (uptrend)
                vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought + weekly downtrend + volume confirmation
            elif (rsi_val > 75 and  # Overbought
                  price_close < weekly_trend and  # Below weekly EMA50 (downtrend)
                  vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI returns to neutral zone or opposite extreme
            if position == 1 and (rsi_val > 50 or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 50 or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_RSI_Extreme_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0