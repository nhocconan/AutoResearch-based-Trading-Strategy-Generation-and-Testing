#!/usr/bin/env python3
"""
Experiment #588: 30m Primary + 4h/1d HTF — Simplified Trend-Pullback with Volume Filter

Hypothesis: After analyzing 500+ failed strategies, the pattern is clear:
- 30m strategies #578, #580, #585 ALL failed with 0 trades (Sharpe=0.000)
- Root cause: TOO MANY filters (chop + CRSI + session + volume = no trades)
- #579 (4h CRSI + 1d HMA) worked because it was SIMPLE with wide RSI bands
- For 30m to work: use 1d/4h for DIRECTION, 30m only for ENTRY TIMING
- Keep filters MINIMAL: HTF trend + RSI pullback + volume (NO session filter)
- Use WIDE RSI bands (25-75) to ensure trades generate on lower TF
- Target: 40-80 trades/year on 30m (per Rule 10), Sharpe > 0.520 to beat current best

This strategy uses PROVEN simple logic from #579 but adapted for 30m:
1. 1d HMA(21) for PRIMARY trend direction (slow, reliable HTF bias)
2. 4h RSI(14) for pullback entries: long when RSI<35 in uptrend, short when RSI>65 in downtrend
3. 30m volume > 0.8x 20-bar avg (minimal volume confirmation)
4. ATR(14) 2.5x trailing stop for all positions
5. Position size: 0.25 (smaller for 30m per Rule 10, max 0.40)

Why this might beat Sharpe=0.520:
- 30m entries within 1d/4h trend = more precise timing than 4h-only
- SIMPLER filters = MORE trades = less chance of 0-trade failure
- Volume filter is MINIMAL (0.8x avg) to not kill trade count
- WIDE RSI bands ensure entries happen during normal pullbacks
- 1d HTF direction is MORE reliable than 4h (proven in #579)

Position sizing: 0.25 base (discrete per Rule 4, smaller for 30m TF)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_hma_4h1d_volume_v1"
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 4h RSI for pullback timing
    rsi_4h_14 = calculate_rsi(df_4h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    rsi_4h_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_14)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m per Rule 10)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_4h_14_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H RSI PULLBACK (wide bands for more trades) ===
        # Long: RSI < 35 in uptrend (oversold pullback)
        # Short: RSI > 65 in downtrend (overbought rally)
        rsi_oversold_long = rsi_4h_14_aligned[i] < 35.0
        rsi_overbought_short = rsi_4h_14_aligned[i] > 65.0
        
        # === VOLUME FILTER (minimal - just confirm activity) ===
        # Volume > 0.8x 20-bar average (very permissive)
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === ENTRY LOGIC — SIMPLE (fewer filters = more trades) ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h RSI oversold + volume OK
        if bull_regime_1d and rsi_oversold_long and volume_ok:
            # Size based on 1d trend strength
            if hma_1d_slope_bull:
                new_signal = POSITION_SIZE
            else:
                new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRY: 1d bear + 4h RSI overbought + volume OK
        elif bear_regime_1d and rsi_overbought_short and volume_ok:
            # Size based on 1d trend strength
            if hma_1d_slope_bear:
                new_signal = -POSITION_SIZE
            else:
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
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
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