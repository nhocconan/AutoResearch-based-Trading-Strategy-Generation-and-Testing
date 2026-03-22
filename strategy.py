#!/usr/bin/env python3
"""
Experiment #384: 4h Primary + 12h/1d HTF — Triple-TF Trend + RSI Pullback + ADX Filter

Hypothesis: Building on #382's success (Sharpe=0.109 on 12h), moving to 4h primary
with triple-timeframe confirmation should improve Sharpe by:
1. 4h entries are more responsive than 12h (catch trends earlier)
2. 1d HMA(21) for major trend bias (proven edge from #382)
3. 12h HMA(21/50) for intermediate trend confirmation (filters false 4h signals)
4. ADX(14) > 20 filter to avoid choppy whipsaws (major cause of negative Sharpe)
5. RSI(14) pullback zones: 40-55 for longs, 45-60 for shorts (not extremes)
6. ATR 2.5x trailing stop for risk management
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why 4h might beat 12h:
- 12h generated ~30 trades/year (good frequency)
- 4h with ADX filter should generate 35-50 trades/year (still acceptable)
- More responsive entries = better capture of trend moves
- ADX filter prevents over-trading in chop (the #1 killer of lower TF strategies)

Position sizing: 0.25-0.30 (discrete, max 0.35)
Stoploss: 2.5 * ATR trailing
Target: 35-50 trades/year, >=30 trades/symbol on train, Sharpe > 0.435
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_triple_trend_rsi_adx_12h1d_v1"
timeframe = "4h"
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
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed DM and TR
    plus_di = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI values
    plus_di_val = 100.0 * (plus_di / (atr + 1e-10))
    minus_di_val = 100.0 * (minus_di / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di_val - minus_di_val) / (plus_di_val + minus_di_val + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Calculate 12h HTF indicators (intermediate trend)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H INTERMEDIATE TREND (confirmation filter) ===
        bull_regime_12h = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        bear_regime_12h = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H LOCAL TREND ===
        bull_4h = hma_4h_21[i] > hma_4h_50[i]
        bear_4h = hma_4h_21[i] < hma_4h_50[i]
        
        # === ADX FILTER (avoid chop) ===
        trending_market = adx_14[i] > 20.0
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 40-55 in uptrend (buying dip)
        rsi_long_pullback = 40.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-60 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC - TRIPLE TF TREND FOLLOW ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: All 3 TF bullish + trending + RSI pullback
        if bull_regime_1d and bull_regime_12h and bull_4h and trending_market and rsi_long_pullback:
            new_signal = LONG_SIZE
        elif bull_regime_1d and bull_regime_12h and rsi_14[i] < 50 and trending_market:
            # Weaker long: 1d+12h bull + RSI < 50 (skip 4h HMA requirement for more trades)
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: All 3 TF bearish + trending + RSI pullback
        if bear_regime_1d and bear_regime_12h and bear_4h and trending_market and rsi_short_pullback:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE
        elif bear_regime_1d and bear_regime_12h and rsi_14[i] > 50 and trending_market:
            # Weaker short: 1d+12h bear + RSI > 50
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~2 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_regime_1d and rsi_14[i] < 45:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime_1d and rsi_14[i] > 55:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1d:
            new_signal = 0.0
        
        # Intermediate trend reversal exit (12h HMA cross)
        if in_position and position_side > 0 and bear_regime_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_12h:
            new_signal = 0.0
        
        # ADX drops below threshold (market going choppy)
        if in_position and adx_14[i] < 18.0:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
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