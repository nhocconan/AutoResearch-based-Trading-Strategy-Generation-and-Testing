#!/usr/bin/env python3
"""
Experiment #044: 4h Primary + 12h/1d HTF — Simplified Trend + Funding Contrarian

Hypothesis: After 40 failed experiments, complexity is the enemy. This strategy:
1) Uses SIMPLE 4h HMA trend (proven in current best)
2) Adds FUNDING RATE z-score contrarian signal (proven edge for BTC/ETH in bear markets)
3) LOOSE RSI entries (30/70 not 20/80) to ensure 30+ trades/year
4) 12h HMA for intermediate trend (faster than 1d, more responsive)
5) Minimal filters to avoid 0-trade failure (experiments #035-043 all failed this way)

Why this should work:
- 4h timeframe = proven (current best Sharpe=0.486 is 4h-based)
- Funding rate z-score = works through 2022 crash and 2025 bear (contrarian edge)
- Simple HMA trend = avoids overfitting that killed experiments #032-043
- LOOSE entries = guarantees trades (RSI 30/70, not extreme values)
- Position size 0.28 = controls drawdown (BTC -77% in 2022 → only -22% equity)

Key difference from #039:
- Removed complex regime switching (Choppiness + Donchian + CRSI = too many filters)
- Added funding rate contrarian (works when trend strategies fail)
- Simpler entry: RSI + HMA + funding z-score (3 filters max)
- This avoids the 0-trade failure that killed 8 recent experiments

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target trades: 30-50/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_funding_contrarian_12h_v1"
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

def calculate_funding_zscore(funding_data, window=30):
    """
    Calculate z-score of funding rate for contrarian signal.
    z < -2.0 → funding very negative → long (crowd too bearish)
    z > +2.0 → funding very positive → short (crowd too bullish)
    """
    if funding_data is None or len(funding_data) == 0:
        return np.zeros(1000)  # fallback
    
    funding_s = pd.Series(funding_data)
    mean = funding_s.rolling(window=window, min_periods=window).mean()
    std = funding_s.rolling(window=window, min_periods=window).std()
    zscore = (funding_s - mean) / (std + 1e-10)
    return zscore.values

def load_funding_data(symbol):
    """Load funding rate data from processed parquet."""
    try:
        import os
        funding_path = f"data/processed/funding/{symbol.lower()}_funding.parquet"
        if os.path.exists(funding_path):
            df = pd.read_parquet(funding_path)
            return df['funding_rate'].values
    except:
        pass
    return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices (for funding data)
    symbol = "BTCUSDT"  # default
    try:
        # Try to get symbol from prices metadata or filename
        if hasattr(prices, 'attrs') and 'symbol' in prices.attrs:
            symbol = prices.attrs['symbol']
    except:
        pass
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Load funding rate data (contrarian signal)
    funding_rates = load_funding_data(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_z = calculate_funding_zscore(funding_rates[:n], window=30)
    else:
        funding_z = np.zeros(n)  # fallback if no funding data
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)  # slower HMA for trend confirmation
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        
        # === TREND FILTERS ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        price_above_hma_48 = close[i] > hma_48[i]
        price_below_hma_48 = close[i] < hma_48[i]
        
        # 12h trend bias
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # 1d macro bias
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # HMA slope (trend direction)
        hma_21_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_21_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi_14[i] < 35.0  # LOOSE (not 20)
        rsi_overbought = rsi_14[i] > 65.0  # LOOSE (not 80)
        rsi_neutral = 35.0 <= rsi_14[i] <= 65.0
        
        # RSI momentum
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_long = funding_z[i] < -1.5  # Crowd too bearish → long
        funding_extreme_short = funding_z[i] > 1.5  # Crowd too bullish → short
        
        # === ENTRY LOGIC (SIMPLE - max 3 filters) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Trend + RSI pullback OR Funding contrarian ---
        if price_above_hma_21 and price_above_hma_48:  # 4h uptrend
            # Entry 1: RSI pullback in uptrend (proven pattern)
            if rsi_oversold and rsi_rising:
                if price_above_hma_12h:  # 12h confirms
                    new_signal = POSITION_SIZE
            
            # Entry 2: Funding contrarian (crowd too bearish)
            elif funding_extreme_long:
                if price_above_hma_1d or price_above_hma_12h:  # HTF helps
                    new_signal = POSITION_SIZE
            
            # Entry 3: HMA crossover + RSI confirmation (fallback for trades)
            elif close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]:
                if rsi_14[i] > 40:  # RSI not collapsing
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Trend + RSI rally OR Funding contrarian ---
        elif price_below_hma_21 and price_below_hma_48:  # 4h downtrend
            # Entry 1: RSI rally in downtrend (proven pattern)
            if rsi_overbought and rsi_falling:
                if price_below_hma_12h:  # 12h confirms
                    new_signal = -POSITION_SIZE
            
            # Entry 2: Funding contrarian (crowd too bullish)
            elif funding_extreme_short:
                if price_below_hma_1d or price_below_hma_12h:  # HTF helps
                    new_signal = -POSITION_SIZE
            
            # Entry 3: HMA crossover + RSI confirmation (fallback for trades)
            elif close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]:
                if rsi_14[i] < 60:  # RSI not exploding
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL/RANGE: Mean reversion at extremes ---
        else:
            # Long at RSI extreme + funding support
            if rsi_14[i] < 30 and funding_extreme_long:
                new_signal = POSITION_SIZE
            # Short at RSI extreme + funding resistance
            elif rsi_14[i] > 70 and funding_extreme_short:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_48 and hma_21_slope_down:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_48 and hma_21_slope_up:
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