# 6H_OrderBlock_Reversal_with_Momentum_Confirmation
# Order block concept: Institutional order blocks identified by strong reversal candles.
# Bullish OB: Last down candle before strong up move (close > open of next 2 candles).
# Bearish OB: Last up candle before strong down move (close < open of next 2 candles).
# Enter on retest of OB with momentum confirmation (RSI divergence or MACD cross).
# Higher timeframe trend filter (12h EMA50) to align with institutional flow.
# Target: 50-150 trades over 4 years with disciplined entries.
# Works in bull/bear: longs in bull trends, shorts in bear trends, avoids chop.

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
    
    # Calculate RSI (14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Identify order blocks: Bullish and Bearish
    bullish_ob_low = np.full(n, np.nan)
    bullish_ob_high = np.full(n, np.nan)
    bearish_ob_low = np.full(n, np.nan)
    bearish_ob_high = np.full(n, np.nan)
    
    # Bullish OB: Down candle followed by strong up move
    # Conditions: close[i-1] < open[i-1] AND close[i] > open[i-1] AND close[i+1] > open[i]
    for i in range(1, n-1):
        if (close[i-1] < high[i-1] and  # Down candle (close < open approximated by close < high)
            close[i] > high[i-1] and    # Strong break above prior high
            i+1 < n and close[i+1] > close[i]):  # Continued strength
            bullish_ob_low[i-1] = low[i-1]
            bullish_ob_high[i-1] = high[i-1]
        
        # Bearish OB: Up candle followed by strong down move
        # Conditions: close[i-1] > open[i-1] AND close[i] < open[i-1] AND close[i+1] < open[i]
        if (close[i-1] > low[i-1] and  # Up candle (close > open approximated by close > low)
            close[i] < low[i-1] and    # Strong break below prior low
            i+1 < n and close[i+1] < close[i]):  # Continued weakness
            bearish_ob_low[i-1] = low[i-1]
            bearish_ob_high[i-1] = high[i-1]
    
    # Align OB levels to 6h timeframe
    bullish_ob_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low, 'high': high}), bullish_ob_low)
    bullish_ob_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low, 'high': high}), bullish_ob_high)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low, 'high': high}), bearish_ob_low)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low, 'high': high}), bearish_ob_high)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if trend filter not ready
        if np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Retest bullish OB in uptrend with RSI > 50 (bullish momentum)
            if (bullish_trend and 
                not np.isnan(bullish_ob_low_aligned[i]) and not np.isnan(bullish_ob_high_aligned[i]) and
                low[i] <= bullish_ob_high_aligned[i] and high[i] >= bullish_ob_low_aligned[i] and  # Price in OB zone
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            
            # Short: Retest bearish OB in downtrend with RSI < 50 (bearish momentum)
            elif (bearish_trend and 
                  not np.isnan(bearish_ob_low_aligned[i]) and not np.isnan(bearish_ob_high_aligned[i]) and
                  low[i] <= bearish_ob_high_aligned[i] and high[i] >= bearish_ob_low_aligned[i] and  # Price in OB zone
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
        
        else:
            # Exit conditions: RSI reversal or OB break
            if position == 1:  # Long position
                # Exit: RSI turns bearish (<40) or price breaks below OB low
                if rsi[i] < 40 or (not np.isnan(bullish_ob_low_aligned[i]) and close[i] < bullish_ob_low_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit: RSI turns bullish (>60) or price breaks above OB high
                if rsi[i] > 60 or (not np.isnan(bearish_ob_high_aligned[i]) and close[i] > bearish_ob_high_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_OrderBlock_Reversal_with_Momentum_Confirmation"
timeframe = "6h"
leverage = 1.0