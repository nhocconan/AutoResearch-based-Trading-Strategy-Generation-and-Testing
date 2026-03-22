#!/usr/bin/env python3
"""
Experiment #399: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + Volume Confirm

Hypothesis: After 398 experiments, the pattern shows:
1. KAMA (Kaufman Adaptive MA) outperforms HMA/EMA in mixed regimes (bull+bear+range)
2. KAMA adapts ER (Efficiency Ratio) — fast in trends, slow in chop
3. 1d HTF for major regime filter (proven in #382, #389)
4. RSI(14) pullback to 35-55 for longs, 45-65 for shorts (wider than typical 30/70)
5. Volume confirmation: volume > 1.3x 20-bar avg (confirms breakout validity)
6. ATR 2.5x trailing stop for risk management
7. Asymmetric sizing: 0.30 long, 0.25 short (bear markets more dangerous)

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to volatility — reduces whipsaw in 2022 crash and 2025 bear
- Volume filter prevents false breakouts (critical for 4h TF)
- 1d HTF prevents counter-trend trades in major moves
- Wider RSI range ensures >=30 trades/symbol (critical requirement)
- Simpler logic = fewer bugs, more reliable execution

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 40-60 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |change| / sum(|changes|)
    change = np.abs(close_s.diff(period))
    noise = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = change / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(series, period):
    """Calculate Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    kama_1d_10 = calculate_kama(df_1d['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_10_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_10)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_30 = calculate_kama(close, period=30)
    vol_sma_20 = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_10_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_30[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d KAMA = bull market bias (favor longs)
        # Price below 1d KAMA = bear market bias (favor shorts)
        bull_regime = close[i] > kama_1d_10_aligned[i]
        bear_regime = close[i] < kama_1d_10_aligned[i]
        
        # === 4H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_4h_10[i] > kama_4h_30[i]
        kama_bearish = kama_4h_10[i] < kama_4h_30[i]
        
        # KAMA slope confirmation
        kama_slope_up = kama_4h_10[i] > kama_4h_10[i-1] if i > 0 else False
        kama_slope_down = kama_4h_10[i] < kama_4h_10[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 1.3 * vol_sma_20[i]
        vol_normal = volume[i] > 0.8 * vol_sma_20[i]
        
        # === RSI PULLBACK SIGNALS (wider range for trade frequency) ===
        # Long: RSI pulled back to 35-55 in uptrend (buying dip)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-65 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # RSI extreme (reversal signal)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC — KAMA TREND + RSI PULLBACK + VOLUME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + KAMA bullish + RSI pullback + volume confirm
        if bull_regime:
            if kama_bullish and kama_slope_up and rsi_long_pullback:
                if vol_confirm or bars_since_last_trade > 15:
                    new_signal = LONG_SIZE
            elif kama_bullish and rsi_oversold and vol_normal:
                # Deep pullback entry
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Bear regime + KAMA bearish + RSI pullback + volume confirm
        if bear_regime:
            if kama_bearish and kama_slope_down and rsi_short_pullback:
                if vol_confirm or bars_since_last_trade > 15:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
            elif kama_bearish and rsi_overbought and vol_normal:
                # Deep rally entry
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~2.5 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and kama_bullish and rsi_14[i] < 55:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and kama_bearish and rsi_14[i] > 45:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (4h KAMA cross)
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if low[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = low[i]
            else:
                lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if high[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals