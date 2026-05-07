# 4h_RSI_Divergence_1dTrend_Volume
# Hypothesis: Uses daily RSI divergence (bullish/bearish) with 4h price action to catch reversals at trend exhaustion.
# Bullish divergence: price makes lower low, RSI makes higher low → long when price breaks above recent swing high.
# Bearish divergence: price makes higher high, RSI makes lower high → short when price breaks below recent swing low.
# Filtered by 1d EMA50 trend and volume confirmation. Works in both bull/bear markets by trading reversals against overextended moves.
# Target: 20-40 trades/year to stay within optimal frequency range.

name = "4h_RSI_Divergence_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate volume spike on 4h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Swing points for divergence detection (5-period lookback)
    swing_low = np.full(len(close), np.nan)
    swing_high = np.full(len(close), np.nan)
    
    for i in range(5, len(close)):
        # Swing low: lowest low in window
        if low[i] == np.min(low[i-5:i+1]):
            swing_low[i] = low[i]
        # Swing high: highest high in window
        if high[i] == np.max(high[i-5:i+1]):
            swing_high[i] = high[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi_4h[i]) or np.isnan(ema_50_1d_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if not np.isnan(swing_low[i]) and i >= 10:
                # Look for two swing lows where second is lower but RSI is higher
                for j in range(i-10, i):
                    if not np.isnan(swing_low[j]) and not np.isnan(swing_low[i]):
                        if swing_low[i] < swing_low[j] and rsi_4h[i] > rsi_4h[j]:
                            bullish_div = True
                            break
            
            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if not np.isnan(swing_high[i]) and i >= 10:
                # Look for two swing highs where second is higher but RSI is lower
                for j in range(i-10, i):
                    if not np.isnan(swing_high[j]) and not np.isnan(swing_high[i]):
                        if swing_high[i] > swing_high[j] and rsi_4h[i] < rsi_4h[j]:
                            bearish_div = True
                            break
            
            # Long: bullish divergence + price above 1d EMA50 + volume spike + break above recent swing high
            if bullish_div and close[i] > ema_50_1d_4h[i] and volume_spike[i]:
                # Confirm break above recent swing high
                recent_high = np.max(high[max(0, i-20):i+1]) if i >= 20 else np.max(high[:i+1])
                if close[i] > recent_high:
                    signals[i] = 0.25
                    position = 1
            # Short: bearish divergence + price below 1d EMA50 + volume spike + break below recent swing low
            elif bearish_div and close[i] < ema_50_1d_4h[i] and volume_spike[i]:
                # Confirm break below recent swing low
                recent_low = np.min(low[max(0, i-20):i+1]) if i >= 20 else np.min(low[:i+1])
                if close[i] < recent_low:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: bearish divergence or price breaks below 1d EMA50
            bearish_div_exit = False
            if not np.isnan(swing_high[i]) and i >= 10:
                for j in range(i-10, i):
                    if not np.isnan(swing_high[j]) and not np.isnan(swing_high[i]):
                        if swing_high[i] > swing_high[j] and rsi_4h[i] < rsi_4h[j]:
                            bearish_div_exit = True
                            break
            
            if bearish_div_exit or close[i] < ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish divergence or price breaks above 1d EMA50
            bullish_div_exit = False
            if not np.isnan(swing_low[i]) and i >= 10:
                for j in range(i-10, i):
                    if not np.isnan(swing_low[j]) and not np.isnan(swing_low[i]):
                        if swing_low[i] < swing_low[j] and rsi_4h[i] > rsi_4h[j]:
                            bullish_div_exit = True
                            break
            
            if bullish_div_exit or close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals