#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume confirmation
# - Long: 1h RSI(14) crosses above 30 (oversold bounce) AND 4h close > 4h EMA(50) (uptrend) AND 1d volume > 1.5x 20-period average
# - Short: 1h RSI(14) crosses below 70 (overbought rejection) AND 4h close < 4h EMA(50) (downtrend) AND 1d volume > 1.5x 20-period average
# - Exit: RSI returns to 50 level or ATR-based stop (1.5 ATR)
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Williams %R alternative considered but RSI provides clearer entry/exit levels for mean reversion

name = "1h_4h_1d_rsi_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d volume SMA(20)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rs), 50, rsi)  # Handle division by zero
    
    # Pre-compute 1h RSI previous value for crossover detection
    rsi_prev = np.roll(rsi, 1)
    rsi_prev[0] = rsi[0]
    
    # Pre-compute ATR for stoploss (1h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(rsi[i]) or np.isnan(rsi_prev[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # RSI values
        rsi_current = rsi[i]
        rsi_previous = rsi_prev[i]
        
        # 4h trend filter: close vs EMA(50)
        trend_up = close_price > ema_50_4h_aligned[i]
        trend_down = close_price < ema_50_4h_aligned[i]
        
        # 1d volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: RSI crosses above 30 (oversold bounce) in uptrend with volume
        if rsi_previous <= 30 and rsi_current > 30 and trend_up and vol_confirm:
            enter_long = True
        
        # Short: RSI crosses below 70 (overbought rejection) in downtrend with volume
        if rsi_previous >= 70 and rsi_current < 70 and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI returns to 50 or ATR-based stop
            exit_long = (rsi_current >= 50) or (close_price <= entry_price - 1.5 * atr_14[i])
        elif position == -1:
            # Exit short if RSI returns to 50 or ATR-based stop
            exit_short = (rsi_current <= 50) or (close_price >= entry_price + 1.5 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals