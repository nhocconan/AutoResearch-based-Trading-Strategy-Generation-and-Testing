#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume regime filter
# - Long: RSI(14) < 30, price > 4h EMA(50), 1d volume > 1.2x 20-day average
# - Short: RSI(14) > 70, price < 4h EMA(50), 1d volume > 1.2x 20-day average
# - Exit: RSI returns to neutral zone (40-60) or opposite extreme
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Combines mean reversion on 1h with trend and volume filters from higher timeframes
# - Works in both bull and bear markets via RSI extremes + trend alignment

name = "1h_4h_1d_rsi_meanrev_trend_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d volume 20-day SMA for regime filter
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute ATR for stoploss (1h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        rsi_current = rsi[i]
        
        # Trend filter: price relative to 4h EMA(50)
        price_above_4h_ema = close_price > ema_50_4h_aligned[i]
        price_below_4h_ema = close_price < ema_50_4h_aligned[i]
        
        # Volume regime: current volume > 1.2x 20-day average (ensures sufficient liquidity/participation)
        volume_regime = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Mean reversion conditions based on RSI extremes
        rsi_oversold = rsi_current < 30
        rsi_overbought = rsi_current > 70
        rsi_neutral_entry = rsi_current > 40 and rsi_current < 60  # For exit logic
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long setup: RSI oversold + price above 4h EMA (bullish alignment) + volume regime
        if rsi_oversold and price_above_4h_ema and volume_regime:
            enter_long = True
        
        # Short setup: RSI overbought + price below 4h EMA (bearish alignment) + volume regime
        if rsi_overbought and price_below_4h_ema and volume_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI returns to neutral or becomes overbought
            exit_long = rsi_current > 40
        elif position == -1:
            # Exit short if RSI returns to neutral or becomes oversold
            exit_short = rsi_current < 60
        
        # Track entry price for potential stoploss (though we use RSI-based exits)
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