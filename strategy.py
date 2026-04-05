#!/usr/bin/env python3
"""
Experiment #7934: 1-hour momentum with 4h/1d trend filter and volume confirmation.
Hypothesis: In ranging markets (2025-2026), momentum on 1h aligned with 4h trend and 1d regime 
captures mean-reversion bounces with controlled risk. Uses 4h for trend direction, 1d for 
volatility regime (low volatility = mean reversion), and 1h for precise entry timing. 
Target: 60-150 total trades over 4 years (15-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7934_1h_momentum_4h_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d volatility regime (ATR percentile)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    volatility_regime = np.where(atr_1d < atr_ma_1d, 1, -1)  # 1=low vol (mean revert), -1=high vol (trend)
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime)
    
    # LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, RSI_PERIOD, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(volatility_regime_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check stoploss
        if position != 0:
            if position == 1 and close[i] < entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and close[i] > entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine regime alignment
        low_volatility = volatility_regime_aligned[i] == 1
        uptrend_4h = trend_4h_aligned[i] == 1
        downtrend_4h = trend_4h_aligned[i] == -1
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Momentum conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry logic: In low volatility, mean reversion; in high volatility, follow trend
        if position == 0:
            # Low volatility regime: mean reversion at RSI extremes
            if low_volatility:
                if rsi_oversold and volume_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                elif rsi_overbought and volume_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
            # High volatility regime: follow 4h trend on momentum
            else:
                if uptrend_4h and rsi[i] > 50 and rsi[i] < RSI_OVERBOUGHT and volume_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                elif downtrend_4h and rsi[i] < 50 and rsi[i] > RSI_OVERSOLD and volume_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
            # Hold position
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals