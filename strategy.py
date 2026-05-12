# BTC/ETH Multi-Timeframe Reversal Strategy
# Target: 1d timeframe with 1h confirmation for reduced false signals
# Strategy: Use 1h momentum (MACD cross) to confirm 1d mean reversion signals (RSI extremes)
# Entry: Long when 1d RSI < 30 and 1h MACD line crosses above signal
# Entry: Short when 1d RSI > 70 and 1h MACD line crosses below signal
# Exit: When 1d RSI returns to neutral zone (40-60) or opposite signal occurs
# Position sizing: 0.25 for controlled risk
# Expected trades: 15-25 per year (~60-100 total over 4 years)

name = "BTCETH_1d_MeanReversion_MACD_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d indicators (primary timeframe)
    # RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h indicators for confirmation (higher timeframe)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 35:
        return np.zeros(n)
    
    # MACD(12,26,9) on 1h
    close_1h = df_1h['close'].values
    ema_12 = pd.Series(close_1h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = pd.Series(close_1h).ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    
    # Align 1h indicators to 1d timeframe
    macd_line_aligned = align_htf_to_ltf(prices, df_1h, macd_line.values)
    signal_line_aligned = align_htf_to_ltf(prices, df_1h, signal_line.values)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1h, macd_hist.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 35 to ensure all indicators are valid
    for i in range(35, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or 
            np.isnan(macd_line_aligned[i]) or 
            np.isnan(signal_line_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG ENTRY: 1d RSI oversold (<30) + 1h MACD bullish crossover
            if (rsi_values[i] < 30 and 
                macd_line_aligned[i] > signal_line_aligned[i] and
                macd_line_aligned[i-1] <= signal_line_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: 1d RSI overbought (>70) + 1h MACD bearish crossover
            elif (rsi_values[i] > 70 and 
                  macd_line_aligned[i] < signal_line_aligned[i] and
                  macd_line_aligned[i-1] >= signal_line_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # LONG EXIT: RSI returns to neutral (>=40) or bearish crossover occurs
            if (rsi_values[i] >= 40 or 
                (macd_line_aligned[i] < signal_line_aligned[i] and 
                 macd_line_aligned[i-1] >= signal_line_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # SHORT EXIT: RSI returns to neutral (<=60) or bullish crossover occurs
            if (rsi_values[i] <= 60 or 
                (macd_line_aligned[i] > signal_line_aligned[i] and 
                 macd_line_aligned[i-1] <= signal_line_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals