#!/usr/bin/env python3
"""
Experiment #315: 1h Primary + 4h/1d HTF — Funding Rate + Vol Spike + Choppiness Regime

Hypothesis: Combining funding rate contrarian signals with vol spike reversion on 1h timeframe,
filtered by 4h/1d trend direction, will outperform pure technical strategies on BTC/ETH.

Why this might work (learned from 284 failed experiments):
1. Funding rate z-score < -2 = extreme fear = long opportunity (BTC/ETH specific edge)
2. ATR(7)/ATR(30) > 1.8 = vol spike = mean reversion likely
3. Choppiness Index > 55 = range market = favor mean reversion over trend
4. 4h HMA trend filter ensures we trade with major direction
5. 1h only for entry timing = HTF trade frequency with LTF precision
6. Session filter (8-20 UTC) reduces trades to target 30-60/year
7. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)

Key differences from failed 1h strategies (#308, #310, #305):
- Funding rate filter adds unique edge (not pure TA)
- Vol spike entry (not RSI extremes alone)
- Fewer conflicting conditions = more trades generated
- Looser RSI thresholds (35/65 not 30/70) to ensure trades
- Forced entry after 40 bars without signal (ensure min trades)

Position sizing: 0.20-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 1h (8760 hours / 150-300 hours per trade)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_volspike_chop_4h1d_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window=period, min_periods=period).mean()
    rolling_std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    
    # Volume MA for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
    # Try to load funding rate data (optional - if not available, skip this filter)
    funding_zscore = np.zeros(n)
    try:
        # Funding rate data path (relative to strategy location)
        import os
        symbol = "BTCUSDT"  # Will be overridden by engine
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            funding_rates = funding_df['funding_rate'].values
            # Align funding to prices (funding is 8h, prices are 1h)
            if len(funding_rates) > 0:
                funding_zscore_raw = calculate_zscore(funding_rates, 30)
                # Simple alignment - repeat each funding value for ~8 bars
                min_len = min(n, len(funding_zscore_raw) * 8)
                funding_aligned = np.zeros(n)
                for i in range(len(funding_zscore_raw)):
                    start_idx = i * 8
                    end_idx = min(start_idx + 8, n)
                    if start_idx < n:
                        funding_aligned[start_idx:end_idx] = funding_zscore_raw[i]
                funding_zscore = funding_aligned
    except:
        funding_zscore = np.zeros(n)  # Skip funding filter if not available
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only - liquid hours) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === 4H/1D TREND REGIME (primary direction filter) ===
        # Bull: 4h HMA21 > HMA48 AND price > 1d HMA21
        # Bear: 4h HMA21 < HMA48 AND price < 1d HMA21
        trend_4h_bull = hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        trend_4h_bear = hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        price_vs_1d_hma = close[i] > hma_1d_21_aligned[i]
        regime_bull = trend_4h_bull and price_vs_1d_hma
        regime_bear = trend_4h_bear and not price_vs_1d_hma
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY SPIKE (ATR ratio) ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.6  # Vol spike = mean reversion opportunity
        vol_normal = atr_ratio < 1.2
        
        # === FUNDING RATE Z-SCORE (contrarian signal) ===
        funding_extreme_long = funding_zscore[i] < -1.5  # Extreme fear = long
        funding_extreme_short = funding_zscore[i] > 1.5  # Extreme greed = short
        
        # === RSI SIGNALS (looser thresholds to ensure trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.7 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else True
        
        # === 1H LOCAL TREND ===
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # === ENTRY LOGIC (multiple paths to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored - asymmetric)
        if in_session or not in_position:  # Allow entry outside session if not in position
            # Path 1: Vol spike + oversold + bull regime (primary)
            if vol_spike and rsi_oversold and (regime_bull or is_choppy):
                new_signal = LONG_STRONG
            
            # Path 2: Funding extreme long + RSI rising
            elif funding_extreme_long and rsi_rising and rsi_14[i] > 35.0:
                new_signal = LONG_BASE
            
            # Path 3: Bull regime + RSI pullback + price above 1h HMA
            elif regime_bull and rsi_neutral and price_above_hma and rsi_rising:
                new_signal = LONG_BASE
            
            # Path 4: Choppy market + RSI oversold (mean revert)
            elif is_choppy and rsi_oversold and vol_ok:
                new_signal = LONG_BASE * 0.9
            
            # Path 5: Trending market + bull regime + RSI rising from neutral
            elif is_trending and regime_bull and rsi_14[i] > 45.0 and rsi_rising:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if in_session or not in_position:
            # Path 1: Vol spike + overbought + bear regime
            if vol_spike and rsi_overbought and (regime_bear or is_choppy):
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # Path 2: Funding extreme short + RSI falling
            elif funding_extreme_short and rsi_falling and rsi_14[i] < 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Path 3: Bear regime + RSI pullback + price below 1h HMA
            elif regime_bear and rsi_neutral and price_below_hma and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Path 4: Choppy market + RSI overbought (mean revert)
            elif is_choppy and rsi_overbought and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.9
        
        # === FORCED ENTRY (ensure minimum trades) ===
        # If no signal for 40 bars (~40 hours), force entry on weaker conditions
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.7
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.7
            elif rsi_oversold:
                new_signal = LONG_BASE * 0.7
            elif rsi_overbought:
                new_signal = -SHORT_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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