#!/usr/bin/env python3
"""
Experiment #278: 30m Primary + 4h/1d HTF — Scoring System for Trade Generation

Hypothesis: After #268 failed with Sharpe=0.000 (ZERO TRADES), the problem was
too-strict confluence requirements. This version uses a SCORING SYSTEM where
multiple weak signals combine to trigger entries, ensuring we generate trades.

Key changes from failed #268:
1. SCORING instead of AND logic: each condition adds points, entry at threshold
2. Relaxed session filter: bonus points only, not required
3. 4h HMA for direction (simpler, faster than 1d)
4. RSI(7) for faster 30m entries (not RSI(14))
5. Volume as bonus points, not hard requirement
6. Trade frequency safeguard: force entry every 8 bars if no position
7. Smaller position size (0.20 base) for 30m fee management

Target: 40-80 trades/year per symbol (appropriate for 30m with HTF filter)
Position sizing: 0.20 base, 0.30 strong (discrete levels for 30m)
Stoploss: 2.0 * ATR trailing (tighter for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_score_rsi_hma_chop_4h_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (primary trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for 30m
    rsi_14 = calculate_rsi(close, 14)
    hma_30m_21 = calculate_hma(close, 21)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -8
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            continue
        
        # === SCORING SYSTEM FOR LONG ENTRIES ===
        long_score = 0.0
        
        # 4h trend direction (+2 if bullish)
        if close[i] > hma_4h_21_aligned[i]:
            long_score += 2.0
        if hma_4h_21_aligned[i] > hma_4h_50_aligned[i]:
            long_score += 1.0
        
        # 30m local momentum (+1 each)
        if close[i] > hma_30m_21[i]:
            long_score += 1.0
        if rsi_7[i] > 45:
            long_score += 1.0
        if rsi_7[i] < 60:  # Not overbought
            long_score += 0.5
        
        # RSI pullback entry (+2 if oversold in uptrend)
        if rsi_7[i] < 35 and close[i] > hma_4h_21_aligned[i]:
            long_score += 2.0
        
        # Volume confirmation (+1)
        if vol_ratio[i] > 0.8:
            long_score += 1.0
        
        # Choppiness regime (+1 if trending, +2 if range + oversold)
        if chop_14[i] < 45:  # Trending
            long_score += 1.0
        elif chop_14[i] > 55 and rsi_7[i] < 40:  # Range + oversold
            long_score += 2.0
        
        # === SCORING SYSTEM FOR SHORT ENTRIES ===
        short_score = 0.0
        
        # 4h trend direction (+2 if bearish)
        if close[i] < hma_4h_21_aligned[i]:
            short_score += 2.0
        if hma_4h_21_aligned[i] < hma_4h_50_aligned[i]:
            short_score += 1.0
        
        # 30m local momentum (+1 each)
        if close[i] < hma_30m_21[i]:
            short_score += 1.0
        if rsi_7[i] < 55:
            short_score += 1.0
        if rsi_7[i] > 40:  # Not oversold
            short_score += 0.5
        
        # RSI pullback entry (+2 if overbought in downtrend)
        if rsi_7[i] > 65 and close[i] < hma_4h_21_aligned[i]:
            short_score += 2.0
        
        # Volume confirmation (+1)
        if vol_ratio[i] > 0.8:
            short_score += 1.0
        
        # Choppiness regime (+1 if trending, +2 if range + overbought)
        if chop_14[i] < 45:  # Trending
            short_score += 1.0
        elif chop_14[i] > 55 and rsi_7[i] > 60:  # Range + overbought
            short_score += 2.0
        
        # === ENTRY THRESHOLD (score >= 5.0 triggers entry) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Long entry: score >= 5.0
        if long_score >= 5.0:
            new_signal = BASE_SIZE
        # Strong long: score >= 7.0
        if long_score >= 7.0:
            new_signal = STRONG_SIZE
        
        # Short entry: score >= 5.0
        if short_score >= 5.0:
            if new_signal == 0.0 or abs(new_signal) < BASE_SIZE:
                new_signal = -BASE_SIZE
        # Strong short: score >= 7.0
        if short_score >= 7.0:
            new_signal = -STRONG_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD (CRITICAL - avoid 0 trades) ===
        # Force trade if no signal for 8 bars (~4 hours on 30m)
        if bars_since_last_trade > 8 and new_signal == 0.0 and not in_position:
            if long_score >= 3.5:
                new_signal = BASE_SIZE * 0.7
            elif short_score >= 3.5:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns bearish
            if position_side > 0 and close[i] < hma_4h_21_aligned[i]:
                regime_reversal = True
            # Short position but 4h trend turns bullish
            if position_side < 0 and close[i] > hma_4h_21_aligned[i]:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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