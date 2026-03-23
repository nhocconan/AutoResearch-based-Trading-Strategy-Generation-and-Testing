#!/usr/bin/env python3
"""
Experiment #077: 1d Primary + 1w HTF — Simplified HMA Trend with RSI Pullback

Hypothesis: Daily timeframe with weekly HTF bias using HMA crossover for trend direction
and RSI pullback for entry timing will generate 30-50 trades/year with Sharpe > 0.486.

Key innovations:
1) 1d primary timeframe — proven higher TF works better (fewer fees, cleaner signals)
2) 1w HMA for macro bias — only trade in direction of weekly trend
3) HMA(21) crossover on 1d for entry timing — simpler than complex regime switching
4) RSI(14) pullback filter — enter on dips in uptrend (RSI 35-55), rallies in downtrend (RSI 45-65)
5) ATR(14) trailing stop — 2.5x ATR to give room for volatility
6) Volume confirmation — volume > 0.8 * SMA(volume, 20) to avoid low-liquidity entries

Why this should work:
- 1d timeframe reduces noise and fee drag (exp #067 was close with Sharpe=-0.004)
- Simpler entry conditions = more trades (avoiding 0-trade problem)
- Weekly HTF prevents counter-trend trades in bear markets
- RSI pullback entries catch better risk/reward than breakouts
- Discrete position sizing (0.30) controls drawdown

Position size: 0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_volume_confirmation(volume, period=20, threshold=0.8):
    """Confirm volume is adequate (volume > threshold * SMA(volume))."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    confirm = volume > (threshold * vol_sma)
    return confirm

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    vol_confirm = calculate_volume_confirmation(volume, period=20, threshold=0.8)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF MACRO BIAS (1w) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === LTF TREND (1d HMA crossover) ===
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-55 in uptrend
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        # Short: RSI rallied to 45-65 in downtrend
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_confirm[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # Long entry: Weekly bullish + Daily HMA bullish + RSI pullback + Volume
        if price_above_hma_1w and hma_bullish and rsi_pullback_long and vol_ok:
            new_signal = POSITION_SIZE
        
        # Short entry: Weekly bearish + Daily HMA bearish + RSI pullback + Volume
        elif price_below_hma_1w and hma_bearish and rsi_pullback_short and vol_ok:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if trend still intact (relaxed RSI for holding)
        if in_position and new_signal == 0.0:
            if position_side > 0 and hma_bullish and price_above_hma_1w:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and hma_bearish and price_below_hma_1w:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if hma_bearish or price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_bullish or price_above_hma_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals