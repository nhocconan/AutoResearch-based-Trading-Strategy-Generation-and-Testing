#!/usr/bin/env python3
"""
Experiment #018: 30m Primary + 4h/1d HTF — Simplified Trend Pullback

Hypothesis: Previous 30m strategies failed due to TOO STRICT confluence (session+volume+CHOP+CRSI = 0 trades).
This strategy uses SIMPLIFIED logic: 4h HMA for trend direction, 30m RSI for pullback timing.
Key insight from failures: lower TF needs FEWER filters, not more. Use HTF for direction, LTF for entry timing only.

Why this should work:
1. 4h HMA(21) = proven trend filter (used in best 4h strategies)
2. 30m RSI pullback (40-60 zone) = catches dips in uptrend, rallies in downtrend
3. ATR vol filter = avoid dead markets (ATR ratio > 0.8)
4. Z-score overlay = avoid extreme extensions
5. NO session filter (killed trades in #008, #015)
6. NO volume filter (killed trades in #008, #015)

Entry conditions (LOOSE enough to generate 30-80 trades/year):
- Long: 4h HMA bullish + 30m RSI 35-55 + price pullback + z-score < 1.5
- Short: 4h HMA bearish + 30m RSI 45-65 + price rally + z-score > -1.5

Position size: 0.25 (conservative for 30m TF)
Stoploss: 2.0*ATR trailing stop
Target: 30-60 trades/year on 30m (fee-efficient per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_v1"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for higher timeframe bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    zscore_20 = calculate_zscore(close, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # 30m HMA for local trend
    hma_30m = calculate_hma(close, period=21)
    
    # ATR ratio for volatility filter (current ATR / 30-bar avg ATR)
    atr_avg_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / (atr_avg_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(zscore_20[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(hma_30m[i]) or np.isnan(atr_ratio[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS (primary direction filter) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (3-bar lookback)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # === 1D TREND CONFIRMATION (higher TF alignment) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 30M LOCAL TREND ===
        hma_30m_slope_bull = hma_30m[i] > hma_30m[i-3] if i >= 3 else False
        hma_30m_slope_bear = hma_30m[i] < hma_30m[i-3] if i >= 3 else False
        
        # === RSI PULLBACK ZONES (not extremes, just pullbacks) ===
        rsi_pullback_long = (rsi_14[i] >= 35) and (rsi_14[i] <= 55)
        rsi_pullback_short = (rsi_14[i] >= 45) and (rsi_14[i] <= 65)
        
        # === Z-SCORE FILTER (avoid extreme extensions) ===
        zscore_not_extended_long = zscore_20[i] < 1.5
        zscore_not_extended_short = zscore_20[i] > -1.5
        
        # === VOLATILITY FILTER (avoid dead markets) ===
        vol_ok = atr_ratio[i] > 0.7
        
        # === SMA200 FILTER (long-term trend) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE enough to generate trades) ===
        new_signal = 0.0
        
        # Long entry: 4h bullish + RSI pullback + not extended + vol ok
        if price_above_hma_4h and hma_4h_slope_bull and rsi_pullback_long:
            if zscore_not_extended_long and vol_ok:
                # Additional confirmation: 1d alignment OR 30m local bullish
                if price_above_hma_1d or hma_30m_slope_bull:
                    new_signal = POSITION_SIZE
        
        # Short entry: 4h bearish + RSI pullback + not extended + vol ok
        if price_below_hma_4h and hma_4h_slope_bear and rsi_pullback_short:
            if zscore_not_extended_short and vol_ok:
                # Additional confirmation: 1d alignment OR 30m local bearish
                if price_below_hma_1d or hma_30m_slope_bear:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_bull:
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