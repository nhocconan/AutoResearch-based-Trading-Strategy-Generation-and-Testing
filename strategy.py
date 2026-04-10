#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w EMA(20) > EMA(50) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1w EMA(20) < EMA(50) AND volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian channel (20) or RSI(14) reaches extreme (70/30) for mean reversion
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian captures structural breaks; 1w EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in both bull and bear markets: breakouts capture strong moves, trend filter prevents counter-trend trades

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(20) vs EMA(50)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish_1w = ema_20_1w > ema_50_1w
    ema_bearish_1w = ema_20_1w < ema_50_1w
    
    # Align HTF indicators to 1d timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute Donchian channels (20) on 1d data
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute RSI(14) on 1d data for exit signals
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50, rsi)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    # RSI conditions for mean reversion exit
    rsi_overbought = rsi > 70
    rsi_oversold = rsi < 30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike[i]) or np.isnan(rsi[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1w bullish trend AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1w bearish trend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: opposite Donchian break OR RSI extreme (mean reversion)
            long_exit = (close[i] < donchian_low[i]) or rsi_overbought[i]
            short_exit = (close[i] > donchian_high[i]) or rsi_oversold[i]
            
            if (position == 1 and long_exit) or (position == -1 and short_exit):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals