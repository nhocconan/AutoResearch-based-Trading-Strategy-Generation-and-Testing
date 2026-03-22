#!/usr/bin/env python3
"""
Experiment #134: 30m HMA Trend + 4h HMA Filter + Vol Spike Mean Reversion + ATR Stop

Hypothesis: Combining proven components for 30m timeframe after learning from 115+ failures:
- 4h HMA(21) for trend direction (from best strategy mtf_4h_kama_1d_hma_adx_atr_v1 Sharpe=0.478)
- 30m HMA(16) for entry timing with less lag than EMA
- ATR(7)/ATR(30) vol spike ratio for mean reversion opportunities (research shows edge)
- Asymmetric entries: long only when 4h bullish, short only when 4h bearish
- 2*ATR trailing stop for capital protection
- Discrete position sizing (0.20, 0.30) to minimize fee churn

Why this might work on 30m when others failed:
- Simpler than failed multi-indicator strategies (#127, #128, #133 all Sharpe < -2.9)
- 4h HMA filter proven in winning strategies (mtf_4h_kama_1d_hma_adx_atr_v1)
- Vol spike mean reversion captures panic/recovery cycles better than pure trend
- Asymmetric logic avoids shorting bull markets and longing bear markets
- Looser entry conditions ensure sufficient trades (failed strategies had 0 trades)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_4h_vol_spike_asymmetric_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI for momentum confirmation."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi = calculate_rsi(close, 14)
    
    # Vol spike ratio: ATR(7)/ATR(30) > 2.0 = extreme volatility
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    mask = atr_30 > 0
    vol_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4H TREND BIAS (asymmetric - only trade with HTF trend) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 30M TREND MOMENTUM ===
        bull_momentum = close[i] > hma_16[i]
        bear_momentum = close[i] < hma_16[i]
        
        # HMA crossover confirmation
        hma_bull_cross = hma_16[i] > hma_48[i]
        hma_bear_cross = hma_16[i] < hma_48[i]
        
        # === VOL SPIKE FILTER ===
        # Vol ratio > 2.0 = panic/extreme vol (mean reversion opportunity)
        # Vol ratio < 1.2 = calm (trend following better)
        vol_spike = vol_ratio[i] > 2.0
        vol_calm = vol_ratio[i] < 1.2
        
        # === RSI MOMENTUM ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (asymmetric: only when 4h bullish) ===
        if bull_trend_4h:
            # Strong: 4h bull + 30m bull + HMA cross + vol calm + RSI not overbought
            if bull_momentum and hma_bull_cross and vol_calm and not rsi_overbought:
                new_signal = SIZE_STRONG
            # Moderate: 4h bull + 30m bull + vol calm
            elif bull_momentum and vol_calm:
                new_signal = SIZE_BASE
            # Vol spike mean reversion: 4h bull but 30m dipped + RSI oversold
            elif vol_spike and not bull_momentum and rsi_oversold:
                new_signal = SIZE_BASE
            # Ensure trades: 4h bull + 30m bull (minimal filter)
            elif bull_momentum:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (asymmetric: only when 4h bearish) ===
        if bear_trend_4h:
            # Strong: 4h bear + 30m bear + HMA cross + vol calm + RSI not oversold
            if bear_momentum and hma_bear_cross and vol_calm and not rsi_oversold:
                new_signal = -SIZE_STRONG
            # Moderate: 4h bear + 30m bear + vol calm
            elif bear_momentum and vol_calm:
                new_signal = -SIZE_BASE
            # Vol spike mean reversion: 4h bear but 30m rallied + RSI overbought
            elif vol_spike and not bear_momentum and rsi_overbought:
                new_signal = -SIZE_BASE
            # Ensure trades: 4h bear + 30m bear (minimal filter)
            elif bear_momentum:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2 * ATR trailing ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals