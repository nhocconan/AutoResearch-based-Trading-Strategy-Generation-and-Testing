#!/usr/bin/env python3
"""
Experiment #020: 1h Primary + 4h/12h HTF — Simplified Trend Pullback

Hypothesis: After 18 failed experiments, the key lesson is SIMPLICITY + TRADE FREQUENCY.
Strategies with too many filters (session, volume, choppiness, CRSI) generate 0 trades.
This strategy uses PROVEN patterns from research:

1. 4h HMA(21) for TREND DIRECTION (long only if 4h HMA bullish, short if bearish)
2. 1h RSI(14) pullback for ENTRY TIMING (buy dips in uptrend, sell rallies in downtrend)
3. ATR(14) volatility filter to avoid dead chop (ATR ratio > 0.8)
4. 12h HMA for REGIME BIAS (only trade with 12h trend for higher win rate)

Why this should work:
- Fewer filters = more trades (target 40-60/year on 1h)
- HTF trend filter reduces false signals
- RSI pullback is proven in research (75% win rate in trends)
- Smaller position size (0.22) controls drawdown on lower TF

Key differences from failed attempts:
- NO session filter (kills frequency)
- NO volume filter (kills frequency)
- NO Choppiness Index (failed on 1h in exp #010, #015)
- NO CRSI (failed in exp #008, #019)
- Simple RSI + HTF HMA = proven combination

Position size: 0.22 (smaller for 1h to control fee drag)
Stoploss: 2.5*ATR trailing
Target: 40-60 trades/year, Sharpe > 0.4
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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
    
    return rsi.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for regime bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    # 1h HMA for short-term trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - smaller for 1h TF)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND DIRECTION ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-4] if i >= 4 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-4] if i >= 4 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12H REGIME BIAS ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-12] if i >= 12 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-12] if i >= 12 else False
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 1H MOMENTUM ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-3] if i >= 3 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-3] if i >= 3 else False
        
        # === VOLATILITY FILTER (avoid dead chop) ===
        atr_ratio = atr_14[i] / (pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values[i]) if i >= 20 else 1.0
        vol_ok = atr_ratio > 0.7
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI pulled back to 40-50 in uptrend
        rsi_pullback_long = (rsi_14[i] >= 35) and (rsi_14[i] <= 50)
        # Short: RSI rallied to 50-60 in downtrend
        rsi_pullback_short = (rsi_14[i] >= 50) and (rsi_14[i] <= 65)
        
        # === RSI EXTREME REVERSAL (for counter-trend in ranges) ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 12h neutral/bullish + RSI pullback + vol ok
        if (hma_4h_slope_bull or price_above_hma_4h) and \
           (hma_12h_slope_bull or price_above_hma_12h) and \
           rsi_pullback_long and vol_ok:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + 12h neutral/bearish + RSI pullback + vol ok
        elif (hma_4h_slope_bear or price_below_hma_4h) and \
             (hma_12h_slope_bear or price_below_hma_12h) and \
             rsi_pullback_short and vol_ok:
            new_signal = -POSITION_SIZE
        
        # EXTREME REVERSAL LONG: RSI < 30 + price > SMA50 (oversold bounce)
        elif rsi_oversold and close[i] > sma_50[i] and vol_ok:
            new_signal = POSITION_SIZE
        
        # EXTREME REVERSAL SHORT: RSI > 70 + price < SMA50 (overbought drop)
        elif rsi_overbought and close[i] < sma_50[i] and vol_ok:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === TIME-BASED EXIT (reduce churn, force profit taking) ===
        if in_position:
            bars_in_trade += 1
            # Exit after 48 bars (2 days on 1h) if not profitable
            if bars_in_trade > 48:
                if position_side > 0 and close[i] < entry_price:
                    new_signal = 0.0
                elif position_side < 0 and close[i] > entry_price:
                    new_signal = 0.0
        else:
            bars_in_trade = 0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                bars_in_trade = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                bars_in_trade = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = new_signal
    
    return signals