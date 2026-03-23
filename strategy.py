#!/usr/bin/env python3
"""
Experiment #044: 4h Primary + 12h/1d HTF — Funding Rate Proxy Mean Reversion

Hypothesis: Research shows funding rate mean reversion has Sharpe 0.8-1.5 through 2022 crash
for BTC/ETH. Since funding data access is limited, I'll use PRICE-BASED PROXIES that capture
the same sentiment extremes: extreme RSI + Bollinger band position + vol spike = crowded trade.

Key innovations:
1. SENTIMENT EXTREME PROXY: RSI(7) < 20 or > 80 + price outside BB(20,2.5) = crowded position
2. VOLATILITY REGIME: ATR(7)/ATR(21) ratio determines if we're in panic (mean revert) or calm (trend)
3. CHOPPINESS FILTER: CHOP(14) > 50 = range (favor mean reversion), < 40 = trend (favor breakout)
4. 12h HMA slope for intermediate trend, 1d HMA for macro bias
5. ASYMMETRIC ENTRY: Easier to enter mean-revert in ranging, harder in trending

Why this should work:
- Captures the same "crowded trade reversal" dynamic as funding rate mean reversion
- 4h TF targets 25-45 trades/year (fee efficient per Rule 10)
- Proven regime-switching logic from current best (Sharpe=0.424)
- Looser entry thresholds than failed experiments to ensure >=10 trades/symbol

Entry conditions (LOOSE for trade generation):
- Long mean-revert: RSI7 < 25 + price < BB_lower + CHOP > 45 OR vol_spike
- Short mean-revert: RSI7 > 75 + price > BB_upper + CHOP > 45 OR vol_spike
- Long trend: RSI7 < 55 + 12h HMA bullish + 1d HMA bullish + CHOP < 45
- Short trend: RSI7 > 45 + 12h HMA bearish + 1d HMA bearish + CHOP < 45

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_sentiment_extreme_chop_regime_12h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider std for extreme detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_bb_percent_b(close, bb_lower, bb_upper):
    """Calculate %B position within Bollinger Bands."""
    bb_range = bb_upper - bb_lower + 1e-10
    pct_b = (close - bb_lower) / bb_range
    return pct_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for trend bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_21 = calculate_atr(high, low, close, period=21)
    
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    bb_pct_b = calculate_bb_percent_b(close, bb_lower, bb_upper)
    
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_7[i]) or np.isnan(atr_21[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if atr_14[i] == 0 or atr_21[i] == 0:
            continue
        
        # === VOLATILITY REGIME ===
        vol_ratio = atr_7[i] / atr_21[i]
        vol_spike = vol_ratio > 1.6  # Elevated vol = mean reversion opportunity
        vol_calm = vol_ratio < 1.2  # Calm vol = trend following OK
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H TREND BIAS ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-5] if i >= 5 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-5] if i >= 5 else False
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 45.0  # Lower threshold for more mean-revert trades
        is_trending = chop_value < 40.0  # Higher threshold for more trend trades
        
        # === SENTIMENT EXTREME PROXY (Funding Rate Substitute) ===
        # RSI extreme + BB extreme = crowded position likely to reverse
        sentiment_extreme_long = rsi_7[i] < 28 and bb_pct_b[i] < 0.05  # Very oversold
        sentiment_extreme_short = rsi_7[i] > 72 and bb_pct_b[i] > 0.95  # Very overbought
        
        sentiment_moderate_long = rsi_7[i] < 35 and bb_pct_b[i] < 0.15
        sentiment_moderate_short = rsi_7[i] > 65 and bb_pct_b[i] > 0.85
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- MEAN REVERSION (Ranging or Vol Spike) ---
        # Long: sentiment extreme + ranging regime OR vol spike
        if sentiment_extreme_long:
            if is_ranging or vol_spike:
                # Easier entry: just need macro not strongly bearish
                if not (hma_12h_slope_bear and price_below_hma_1d):
                    new_signal = POSITION_SIZE
        
        # Short: sentiment extreme + ranging regime OR vol spike
        elif sentiment_extreme_short:
            if is_ranging or vol_spike:
                # Easier entry: just need macro not strongly bullish
                if not (hma_12h_slope_bull and price_above_hma_1d):
                    new_signal = -POSITION_SIZE
        
        # --- TREND FOLLOWING (Low Chop, Calm Vol) ---
        elif is_trending and vol_calm:
            # Long: moderate sentiment + 12h bullish + 1d confirmation
            if sentiment_moderate_long and hma_12h_slope_bull:
                if price_above_hma_1d or price_above_hma_12h:
                    new_signal = POSITION_SIZE
            
            # Short: moderate sentiment + 12h bearish + 1d confirmation
            elif sentiment_moderate_short and hma_12h_slope_bear:
                if price_below_hma_1d or price_below_hma_12h:
                    new_signal = -POSITION_SIZE
        
        # --- HOLD POSITION LOGIC ---
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
        
        # === EXIT ON STRONG REGIME CHANGE ===
        # Exit long if strong bearish trend emerges
        if in_position and position_side > 0:
            if hma_12h_slope_bear and price_below_hma_1d and is_trending:
                new_signal = 0.0
        
        # Exit short if strong bullish trend emerges
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_1d and is_trending:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals