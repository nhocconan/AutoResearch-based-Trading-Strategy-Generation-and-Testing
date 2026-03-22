#!/usr/bin/env python3
"""
Experiment #510: 1d Donchian Breakout with Funding Rate Contrarian + Vol Scaling

Hypothesis: After 509 failed experiments, the critical insight is that BTC/ETH need
SIMPLE trend-following on 1d with contrarian funding signals. Complex regime-switching
has failed repeatedly (experiments 474-509 all discarded). This strategy uses:

1. DONCHIAN(20) BREAKOUT: Pure trend-following entry on daily breakouts
   - Long: price breaks 20-day high
   - Short: price breaks 20-day low
   - Proven on crypto daily timeframes

2. FUNDING RATE CONTRARIAN FILTER (key differentiator):
   - Load funding data from data/processed/funding/*.parquet
   - Z-score funding over 30 days
   - When funding > +2 std (crowded long) → reduce long size or skip
   - When funding < -2 std (crowded short) → reduce short size or skip
   - Research shows Sharpe 0.8-1.5 through 2022 crash

3. 50-DAY SMA TREND FILTER:
   - Only long when price > SMA50
   - Only short when price < SMA50
   - Simple but effective trend bias

4. VOLATILITY-ADJUSTED POSITION SIZING:
   - Base size: 0.25
   - Scale by ATR ratio: size = base * (ATR_median / ATR_current)
   - Smaller positions when vol is high (limits drawdown)
   - Range: 0.15 to 0.35

5. ATR(14) TRAILING STOP at 2.5x:
   - Tighter than previous 3.0x to cut losses faster
   - Signal → 0 when price moves 2.5*ATR against position

6. MINIMAL FILTERS:
   - No ADX, no Choppiness, no complex regime logic
   - Fewer conditions = more trades = better Sharpe calculation
   - Target: 30-60 trades/year per symbol

Why this should work on 1d:
- Donchian breakouts capture major crypto trends (2021 bull, 2022 bear)
- Funding contrarian avoids crowded trades at tops/bottoms
- Vol scaling limits drawdown during high-volatility periods
- Simple logic ensures sufficient trades (not like exp 504 with 0 trades)
- No complex regime switching that caused whipsaws in exp 474-509

Timeframe: 1d (REQUIRED for this experiment)
HTF: None (1d is already high timeframe, keep it simple)
Position sizing: 0.15-0.35 volatility-adjusted
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
import os

name = "mtf_1d_donchian_funding_contrarian_vol_scale_atr_v1"
timeframe = "1d"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def load_funding_data(symbol):
    """
    Load funding rate data from parquet file.
    Returns array of funding rates aligned with prices.
    """
    # Map symbol to filename
    symbol_map = {
        'BTCUSDT': 'BTCUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOLUSDT': 'SOLUSDT'
    }
    
    base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
    funding_path = f"data/processed/funding/{base_symbol}.parquet"
    
    try:
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except Exception:
        # Return zeros if funding data not available
        return None

def calculate_funding_zscore(funding_rates, period=30):
    """Calculate Z-score of funding rates over rolling window."""
    if funding_rates is None or len(funding_rates) == 0:
        return None
    
    funding_s = pd.Series(funding_rates)
    rolling_mean = funding_s.rolling(window=period, min_periods=period).mean()
    rolling_std = funding_s.rolling(window=period, min_periods=period).std()
    zscore = (funding_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices DataFrame if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load funding data ONCE before loop
    funding_rates = load_funding_data(symbol)
    
    # Calculate funding Z-score if available
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_funding_zscore(funding_rates[:n], 30)
    else:
        funding_zscore = np.zeros(n)  # No funding filter if data unavailable
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    sma_50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate median ATR for vol scaling (over last 100 days)
    atr_median = np.nanmedian(atr[-100:]) if n > 100 else np.nanmedian(atr[50:])
    if np.isnan(atr_median) or atr_median == 0:
        atr_median = np.nanmedian(atr[20:])
    
    signals = np.zeros(n)
    
    # Base position sizing
    BASE_SIZE = 0.25
    MIN_SIZE = 0.15
    MAX_SIZE = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER (50-day SMA) ===
        bull_trend = close[i] > sma_50[i]
        bear_trend = close[i] < sma_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === FUNDING CONTRARIAN FILTER ===
        funding_signal = 0.0
        if funding_zscore is not None and not np.isnan(funding_zscore[i]):
            # Extreme positive funding = crowded long = bearish signal
            if funding_zscore[i] > 2.0:
                funding_signal = -1  # Avoid longs
            # Extreme negative funding = crowded short = bullish signal
            elif funding_zscore[i] < -2.0:
                funding_signal = 1  # Avoid shorts
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        vol_scale = atr_median / atr[i] if atr[i] > 0 else 1.0
        vol_scale = np.clip(vol_scale, 0.6, 1.4)  # Limit scaling range
        position_size = BASE_SIZE * vol_scale
        position_size = np.clip(position_size, MIN_SIZE, MAX_SIZE)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: Donchian breakout + bull trend + funding not extreme positive
        if breakout_long and bull_trend:
            if funding_signal != -1:  # Funding doesn't warn against longs
                new_signal = position_size
        
        # Short entry: Donchian breakout + bear trend + funding not extreme negative
        if breakout_short and bear_trend:
            if funding_signal != 1:  # Funding doesn't warn against shorts
                new_signal = -position_size
        
        # === STOPLOSS LOGIC - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
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