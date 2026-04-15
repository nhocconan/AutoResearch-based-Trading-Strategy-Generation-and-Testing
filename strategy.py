# 1d_Triple_RSI_WeeklyTrend_Reversion
# Hypothesis: On daily timeframe, extreme weekly RSI indicates overextended conditions likely to revert.
# Long when weekly RSI < 30 (oversold) and daily RSI < 40 (momentum aligning).
# Short when weekly RSI > 70 (overbought) and daily RSI > 60 (momentum aligning).
# Uses volume confirmation to avoid false signals in low-liquidity periods.
# Weekly trend filter ensures we trade with the higher timeframe momentum when extreme.
# Designed for low trade frequency (<15/year) to minimize fee drag in ranging/bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_daily = 100 - (100 / (1 + rs))
    rsi_daily = rsi_daily.fillna(50).values
    
    # Weekly RSI (14) from 1w data - calculated once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    delta_w = pd.Series(close_1w).diff()
    gain_w = delta_w.clip(lower=0)
    loss_w = -delta_w.clip(upper=0)
    avg_gain_w = gain_w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_w = loss_w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_w = avg_gain_w / avg_loss_w.replace(0, np.nan)
    rsi_weekly_raw = 100 - (100 / (1 + rs_w))
    rsi_weekly_raw = rsi_weekly_raw.fillna(50).values
    # Align weekly RSI to daily timeframe (waits for weekly close)
    rsi_weekly = align_htf_to_ltf(prices, df_1w, rsi_weekly_raw)
    
    # Volume confirmation: current > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_daily[i]) or np.isnan(rsi_weekly[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Weekly oversold + daily momentum aligning + volume confirmation
        if (rsi_weekly[i] < 30 and rsi_daily[i] < 40 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Weekly overbought + daily momentum aligning + volume confirmation
        elif (rsi_weekly[i] > 70 and rsi_daily[i] > 60 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and rsi_daily[i] >= 40) or
               (signals[i-1] == -0.25 and rsi_daily[i] <= 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Triple_RSI_WeeklyTrend_Reversion"
timeframe = "1d"
leverage = 1.0