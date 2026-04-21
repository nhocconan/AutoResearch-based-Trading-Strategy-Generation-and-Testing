#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly trend filter (EMA34), daily RSI mean-reversion, and volume confirmation.
# In bull markets: buy pullbacks to RSI<40 when weekly trend is up.
# In bear markets: sell rallies to RSI>60 when weekly trend is down.
# Uses volume > 1.2x 20-day average for confirmation. Targets 10-25 trades/year.
# Weekly trend avoids counter-trend trades; RSI provides mean-reversion entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily RSI(14)
    delta = pd.Series(prices['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Daily volume MA(20)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # start after EMA warmup
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above/below EMA34
        weekly_uptrend = prices['close'].iloc[i] > ema_34_1w_aligned[i]
        weekly_downtrend = prices['close'].iloc[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = prices['volume'].iloc[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            if weekly_uptrend and volume_confirm:
                # Bullish weekly trend: buy on RSI dip (mean reversion)
                if rsi[i] < 40:
                    signals[i] = 0.25
                    position = 1
            elif weekly_downtrend and volume_confirm:
                # Bearish weekly trend: sell on RSI rally (mean reversion)
                if rsi[i] > 60:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: RSI mean reversion complete or trend change
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral or weekly trend turns bearish
                if rsi[i] >= 50 or not weekly_uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral or weekly trend turns bullish
                if rsi[i] <= 50 or not weekly_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA34Trend_RSIMeanRev_Volume"
timeframe = "1d"
leverage = 1.0