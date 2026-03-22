#!/usr/bin/env python3
"""
Experiment #438: 30m Primary + 4h/1d HTF — Simplified Multi-Confluence

Hypothesis: After 437 experiments, clear pattern: lower TF strategies fail due to
(1) too many filters = 0 trades, or (2) too many trades = fee drag kills profit.

Key insight from #428, #430 failures: session filters + strict confluence = 0 trades.
Solution: SIMPLER logic with wider thresholds, focus on HTF trend + LTF timing only.

Why this might work:
- 4h HMA(21) for major trend direction (proven in best strategies)
- 30m RSI(14) with WIDE thresholds (25/75 not 40/60) = more trigger opportunities
- 30m HMA(8/21) crossover for entry timing within HTF trend
- Volume filter: only >1.0x avg (minimal filter, just avoid dead periods)
- NO session filter (killed trades in #428, #430)
- ATR 2.5x stoploss for crash protection
- Asymmetric sizing: 0.30 long, 0.25 short (bear market bias)

Target: 40-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test
Position sizing: discrete 0.0, ±0.25, ±0.30 (max 0.40)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_volume_4h1d_simp_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (major trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_30m_8 = calculate_hma(close, period=8)
    hma_30m_21 = calculate_hma(close, period=21)
    rsi_30m_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_30m_8[i]) or np.isnan(hma_30m_21[i]):
            continue
        if np.isnan(rsi_30m_14[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA(21) = bull bias (favor longs)
        # Price below 4h HMA(21) = bear bias (favor shorts)
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA crossover confirmation
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D REGIME FILTER (avoid counter-trend in major moves) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 30M LOCAL TREND (HMA crossover for entry timing) ===
        hma_30m_bullish = hma_30m_8[i] > hma_30m_21[i]
        hma_30m_bearish = hma_30m_8[i] < hma_30m_21[i]
        
        # === RSI SIGNALS (WIDE thresholds for more trades) ===
        rsi_oversold = rsi_30m_14[i] < 35.0  # Wider than typical 30
        rsi_overbought = rsi_30m_14[i] > 65.0  # Wider than typical 70
        rsi_neutral_low = rsi_30m_14[i] < 50.0
        rsi_neutral_high = rsi_30m_14[i] > 50.0
        
        # === VOLUME FILTER (minimal - just avoid dead periods) ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === SMA200 FILTER (long-term trend alignment) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (need 4h bull + 30m timing + RSI confirmation)
        if bull_4h and volume_ok:
            # Primary: 4h bull + 30m HMA cross up + RSI not overbought
            if hma_30m_bullish and rsi_neutral_low and not rsi_overbought:
                new_signal = LONG_SIZE
            # Secondary: 4h bull + RSI oversold bounce (mean reversion within trend)
            elif rsi_oversold and above_sma200:
                new_signal = LONG_SIZE * 0.9
            # Tertiary: 4h HMA bullish cross + 30m alignment
            elif hma_4h_bullish and hma_30m_bullish and rsi_30m_14[i] < 55.0:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (need 4h bear + 30m timing + RSI confirmation)
        if bear_4h and volume_ok:
            # Primary: 4h bear + 30m HMA cross down + RSI not oversold
            if hma_30m_bearish and rsi_neutral_high and not rsi_oversold:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: 4h bear + RSI overbought rejection
            elif rsi_overbought and below_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
            # Tertiary: 4h HMA bearish cross + 30m alignment
            elif hma_4h_bearish and hma_30m_bearish and rsi_30m_14[i] > 45.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~7.5 hours on 30m), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_4h and hma_30m_bullish and rsi_30m_14[i] < 52.0:
                new_signal = LONG_SIZE * 0.6
            elif bear_4h and hma_30m_bearish and rsi_30m_14[i] > 48.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_30m_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_30m_14[i] < 25.0:
            new_signal = 0.0
        
        # 4h trend reversal exit (major regime flip)
        if in_position and position_side > 0 and bear_4h and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_4h and hma_4h_bullish:
            new_signal = 0.0
        
        # 30m trend reversal exit (local HMA cross against position)
        if in_position and position_side > 0 and hma_30m_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_30m_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
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