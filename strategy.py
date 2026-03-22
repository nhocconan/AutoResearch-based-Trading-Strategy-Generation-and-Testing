#!/usr/bin/env python3
"""
Experiment #535: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 479 failed strategies (mostly complex regime/choppiness/volspike combos),
return to a PROVEN pattern: HTF trend direction + LTF pullback entries.

Key insights from failures:
- Choppiness Index regimes: FAILED repeatedly (Sharpe -2.5 to -3.0)
- Complex multi-filter entries: Often 0 trades or negative Sharpe
- Volspike strategies: Consistently failed across 20+ variants
- SIMPLE trend + pullback works best (current best Sharpe=0.435)

This strategy uses:
1. 1d HMA(21) for MAJOR trend direction — only trade WITH HTF trend
2. 4h HMA(16/48) for intermediate trend confirmation
3. 1h RSI(14) pullback entries — enter on dips in uptrend, rallies in downtrend
4. Volume filter (>0.8x 20-bar avg) — avoid low liquidity entries
5. Session filter (8-20 UTC) — trade during high liquidity hours
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete position sizing (0.25) to minimize fee churn

Why this might work for 1h:
- 1d/4h HTF filters reduce trade frequency to target 30-60/year
- RSI pullback entries catch better prices than breakout chasing
- Session filter avoids overnight whipsaw and low liquidity
- Simple logic = consistent signals across BTC/ETH/SOL
- 1h TF with HTF direction = proven pattern (research note #7)

Position sizing: 0.25 (conservative for lower TF, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_v1"
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

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 4h HTF HMA for intermediate trend
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Extract UTC hour for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
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
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        if vol_sma_20[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bull = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bear = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        session_ok = (utc_hours[i] >= 8) and (utc_hours[i] <= 20)
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI pulled back but not oversold (35-50 range in uptrend)
        rsi_pullback_long = (rsi_14[i] >= 35.0) and (rsi_14[i] <= 50.0)
        # Short: RSI rallied but not overbought (50-65 range in downtrend)
        rsi_pullback_short = (rsi_14[i] >= 50.0) and (rsi_14[i] <= 65.0)
        
        # RSI oversold/overbought for stronger signals
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC — HTF TREND + LTF PULLBACK ===
        new_signal = 0.0
        
        # LONG ENTRIES (only in bull regime with pullback)
        if bull_regime and volume_ok and session_ok:
            # Condition 1: 1d bull + 4h bull + RSI pullback (primary entry)
            if hma_slope_bull and hma_4h_bull and rsi_pullback_long:
                new_signal = POSITION_SIZE
            # Condition 2: 1d bull + RSI oversold (deep pullback entry)
            elif hma_slope_bull and rsi_oversold:
                new_signal = POSITION_SIZE
            # Condition 3: 1d bull + 4h bull + RSI crossing up from oversold
            elif bull_regime and hma_4h_bull and rsi_14[i] > 40.0 and rsi_14[i-1] <= 40.0:
                new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (only in bear regime with pullback)
        if new_signal == 0.0 and bear_regime and volume_ok and session_ok:
            # Condition 1: 1d bear + 4h bear + RSI pullback (primary entry)
            if hma_slope_bear and hma_4h_bear and rsi_pullback_short:
                new_signal = -POSITION_SIZE
            # Condition 2: 1d bear + RSI overbought (bounce entry)
            elif hma_slope_bear and rsi_overbought:
                new_signal = -POSITION_SIZE
            # Condition 3: 1d bear + 4h bear + RSI crossing down from overbought
            elif bear_regime and hma_4h_bear and rsi_14[i] < 60.0 and rsi_14[i-1] >= 60.0:
                new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif rsi_14[i] > 75.0:  # Extreme overbought - take profit
                new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif rsi_14[i] < 25.0:  # Extreme oversold - take profit
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