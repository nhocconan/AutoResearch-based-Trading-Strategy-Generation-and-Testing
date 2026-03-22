#!/usr/bin/env python3
"""
Experiment #570: 1h Primary + 4h/12h HTF — Ultra-Simplified Trend-Pullback

Hypothesis: After 500+ failed experiments, the pattern is CLEAR:
- Complex filters (session, volume, chop, ADX, Fisher) = NEGATIVE Sharpe or 0 trades
- #555 had Sharpe=0.000 with simplified logic but still failed
- #565 (1h hma rsi vol) had Sharpe=-0.427 — volume filter added no edge
- Recent 4h/12h/1d strategies ALL negative: -0.15 to -4.2 Sharpe

NEW APPROACH for 1h:
1. Use ONLY 4h HMA(21) for trend direction (proven, simple)
2. Add 12h HMA(50) as MAJOR regime filter (only trade WITH 12h trend)
3. 1h RSI(14) pullback: LONG when RSI 35-50 in bull, SHORT when RSI 50-65 in bear
4. NO ADX filter (kills trades, no edge per #555, #569)
5. NO volume filter (failed in #565)
6. NO session filter (kills trade frequency)
7. ATR(14) 2.5x trailing stop
8. Position size: 0.25 discrete (smaller for 1h per Rule 10)

Why this might work when others failed:
- 12h HMA filter prevents trading against MAJOR trend (2022 crash protection)
- 4h HMA for entry timing within 12h regime
- RSI pullback catches dips/rallies (mean reversion within trend)
- SIMPLE = less overfitting, more robust across BTC/ETH/SOL
- Target: 40-80 trades/year on 1h (Rule 10), not 200+ which causes fee drag

Key difference from #555:
- Added 12h HMA(50) as major regime filter (was missing)
- Narrower RSI bands (35-50 long, 50-65 short) for better entries
- Both 4h AND 12h must agree on trend direction

Position sizing: 0.25 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_ultra_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF HMA for intermediate trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h HTF HMA for major regime
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 1h vs 4h (Rule 10: lower TF = more trades = smaller size)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 12H MAJOR REGIME (primary filter - only trade WITH major trend) ===
        # 12h HMA21 > HMA50 = major bull regime
        # 12h HMA21 < HMA50 = major bear regime
        major_bull_12h = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        major_bear_12h = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # Price relative to 12h HMA21 for extra confirmation
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND (entry timing within 12h regime) ===
        # 4h HMA21 > HMA50 = intermediate bull
        # 4h HMA21 < HMA50 = intermediate bear
        inter_bull_4h = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        inter_bear_4h = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Price relative to 4h HMA21
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === RSI PULLBACK ENTRY (within trend) ===
        # LONG: RSI 35-50 (pullback, not crash) in bull regime
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        # SHORT: RSI 50-65 (rally into resistance) in bear regime
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — BOTH 12h AND 4h MUST AGREE ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h bull + RSI pullback
        # Price can be below 4h HMA (pullback) but above 12h HMA (major support)
        if major_bull_12h and inter_bull_4h and rsi_pullback_long:
            # Extra confirmation: price above 12h HMA (major support holding)
            if price_above_12h_hma:
                new_signal = POSITION_SIZE
            # Or price pulled back to 12h HMA but 4h still bull
            elif price_below_12h_hma and price_above_4h_hma:
                new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRY: 12h bear + 4h bear + RSI pullback
        # Price can be above 4h HMA (rally) but below 12h HMA (major resistance)
        elif major_bear_12h and inter_bear_4h and rsi_pullback_short:
            # Extra confirmation: price below 12h HMA (major resistance holding)
            if price_below_12h_hma:
                new_signal = -POSITION_SIZE
            # Or price rallied to 12h HMA but 4h still bear
            elif price_above_12h_hma and price_below_4h_hma:
                new_signal = -POSITION_SIZE * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 12h regime flip to bear (major trend change)
        if in_position and position_side > 0:
            if major_bear_12h:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull (major trend change)
        if in_position and position_side < 0:
            if major_bull_12h:
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
                # Flip position
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