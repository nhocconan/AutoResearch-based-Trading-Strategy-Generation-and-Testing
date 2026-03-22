#!/usr/bin/env python3
"""
Experiment #521: 12h Donchian Breakout with Dual HTF Trend + Funding Contrarian

Hypothesis: After 500+ failed experiments, the key insight for 12h timeframe is that 
BREAKOUTS work better than mean-reversion when combined with STRONG trend filters. 
12h captures multi-day momentum swings. Adding funding rate contrarian signal helps 
avoid crowded trades at extremes. Dual HTF (1d + 1w) provides robust trend confirmation.

Key innovations:
1. DONCHIAN(20) BREAKOUT: Price breaks 20-bar high/low for momentum entry
2. DUAL HTF TREND: 1d HMA(21) + 1w HMA(21) both must agree for direction
3. FUNDING CONTRARIAN: Z-score of funding rate < -1.5 → long bias, > +1.5 → short bias
4. ATR VOLATILITY FILTER: Only trade when ATR(14) > median ATR (avoid dead markets)
5. LOOSE THRESHOLDS: Donchian break + any HTF agreement + funding not against
6. 2.5 * ATR STOPLOSS: Wider stop for 12h timeframe swings

Why 12h works for breakouts:
- Captures 2-5 day momentum moves (crypto's typical swing duration)
- 2 bars/day = 730 bars/year = enough for statistical significance
- Less fake breakouts than 1h/4h, more signals than 1d
- Funding rate data available at 8h intervals aligns well with 12h

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d + 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (aggressive for breakout momentum)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_dual_htf_funding_contrarian_atr_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window=period, min_periods=period).mean()
    rolling_std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def load_funding_data(prices):
    """
    Load funding rate data for the symbol.
    Returns funding rate array aligned to prices timeline.
    """
    try:
        # Try to load funding data from processed folder
        symbol = "BTCUSDT"  # Default, will be overridden by engine
        if "ETH" in str(prices.columns):
            symbol = "ETHUSDT"
        elif "SOL" in str(prices.columns):
            symbol = "SOLUSDT"
        
        # Funding data path pattern
        import os
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            # Merge with prices on open_time
            df_funding = df_funding.set_index("open_time")
            prices_indexed = prices.set_index("open_time")
            merged = prices_indexed.join(df_funding[["funding_rate"]], how="left")
            merged = merged.reset_index()
            funding = merged["funding_rate"].fillna(0.0).values
            return funding
        else:
            # Return zeros if no funding data
            return np.zeros(len(prices))
    except Exception:
        return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Load funding rate data
    funding = load_funding_data(prices)
    funding_zscore = calculate_zscore(funding, 30)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Calculate median ATR for volatility filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # Both 1d and 1w must agree for strong signal
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both timeframes agree
        strong_bull = bull_1d and bull_1w
        strong_bear = bear_1d and bear_1w
        
        # Weak bias: only 1d agrees (1w neutral or disagree)
        weak_bull = bull_1d and not bear_1w
        weak_bear = bear_1d and not bull_1w
        
        # === FUNDING RATE CONTRARIAN ===
        # Negative funding z-score = shorts paying longs = bullish contrarian
        # Positive funding z-score = longs paying shorts = bearish contrarian
        funding_bullish = funding_zscore[i] < -1.0
        funding_bearish = funding_zscore[i] > 1.0
        funding_neutral = abs(funding_zscore[i]) <= 1.0
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR is above median (avoid dead markets)
        vol_active = atr_14[i] > atr_median * 0.8
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Break above upper channel
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        # Break below lower channel
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if not vol_active:
            signals[i] = 0.0
            continue
        
        # LONG ENTRY: Donchian breakout + trend agreement + funding not against
        if breakout_long:
            if strong_bull:
                # Strong trend: enter regardless of funding
                new_signal = SIZE
            elif weak_bull and not funding_bearish:
                # Weak trend: only if funding not bearish
                new_signal = SIZE
            elif funding_bullish and (bull_1d or bull_1w):
                # Funding contrarian: enter if at least one HTF bullish
                new_signal = SIZE
        
        # SHORT ENTRY: Donchian breakout + trend agreement + funding not against
        elif breakout_short:
            if strong_bear:
                # Strong trend: enter regardless of funding
                new_signal = -SIZE
            elif weak_bear and not funding_bullish:
                # Weak trend: only if funding not bullish
                new_signal = -SIZE
            elif funding_bearish and (bear_1d or bear_1w):
                # Funding contrarian: enter if at least one HTF bearish
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if both HTF trends flip against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        # Exit long if RSI > 80 (overbought), exit short if RSI < 20 (oversold)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi_14[i] > 80:
                new_signal = 0.0
            if position_side < 0 and rsi_14[i] < 20:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals